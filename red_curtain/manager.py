"""Single-operation model lifecycle. No downloads and no inference fallback."""
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import threading
import time
import uuid
import subprocess
import sys
import torch
import psutil
import soundfile as sf

from red_curtain.adapter import DramaAdapter
from red_curtain.safety import Monitor
from red_curtain.safety import gpu_snapshot
from red_curtain.config import ConfigStore
from red_curtain.voices import VoiceLibrary
from red_curtain.audio import audio_activity
from red_curtain.gpu import gpu_identity,device_memory
from red_curtain.budget import allocator_ceiling,GIB

logger = logging.getLogger(__name__)
TESTED = Path(__file__).with_name('tested.json')


class ModelManager:
    def __init__(self, root, adapter_factory=DramaAdapter, monitor_factory=Monitor):
        self.root = Path(root).resolve()
        self.tested = TESTED
        self.config=ConfigStore(self.root)
        self.outputs = self.config.resolve(self.config.value.output_directory)
        self.voices=VoiceLibrary(self.root)
        self.resources={}
        self.resources_time=0
        self.unload_metrics={}
        self.activity_cache={}
        self.record_cache={}
        self.factory, self.monitor_factory = adapter_factory, monitor_factory
        self.executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix='drama-inference')
        self.lock = threading.RLock()
        self.cancel_event = threading.Event()
        self.adapter = self.monitor = None
        self.state = 'unloaded'
        self.message = 'No model loaded.'
        self.error = self.model_path = self.job = None
        self.stage='idle'
        self.future = None

    def status(self):
        now=time.monotonic()
        if self.monitor and self.monitor.samples:
            self.resources=self.monitor.samples[-1]
        elif now-self.resources_time>3:
            self.resources_time=now
            try:self.resources={'time':time.time(),'gpu':gpu_snapshot(),
                                'process_rss_bytes':psutil.Process().memory_info().rss}
            except Exception as exc:self.resources={'error':str(exc)}
            self.resources_time=now
        hardware=dict(gpu_identity(0))
        if hardware['available']:
            try:
                memory=device_memory(0)
                hardware.update(free_vram_bytes=memory['free_bytes'],
                    suggested_vram_limit_gib=allocator_ceiling(**memory)/GIB)
            except Exception as exc:hardware['memory_error']=str(exc)
        with self.lock:
            return dict(state=self.state,stage=self.stage,message=self.message,error=self.error,
                        model_path=self.model_path,job=self.job,
                        compatibility=self.compatibility(self.model_path) if self.model_path else 'unloaded',
                        capabilities=dict(voice_cloning=True,int4=False,cpu_expert_offload=True),
                        resources=self.resources,
                        hardware=hardware,
                        memory_budget=getattr(self.adapter,'memory_budget',None),
                        unload_metrics=self.unload_metrics,
                        metrics=dict(self.adapter.metrics) if self.adapter else {})

    def compatibility(self,path):
        try:
            raw=json.loads((Path(path)/'config.json').read_text())
            if raw.get('quantization_config',{}).get('quant_method')=='drama3-int4':return 'unsupported'
            evidence=json.loads(self.tested.read_text())
            provenance=json.loads((Path(path)/'download-provenance.json').read_text())
            hardware=gpu_identity(0)
            if (provenance.get('revision')==evidence['model_revision'] and hardware['available']
                and hardware['name']==evidence.get('gpu') and hardware['architecture']==evidence.get('architecture')):
                return 'tested_rocm'
        except (OSError,ValueError,KeyError,TypeError):pass
        return 'experimental_unverified'

    def save_settings(self,value):
        with self.lock:
            if self.state!='unloaded':raise RuntimeError('Unload the model before changing settings')
            self.config.save(value)
            self.outputs=self.config.resolve(value.output_directory)

    def download(self):
        with self.lock:
            self._begin('downloading',{'unloaded','error'})
            self.future=self.executor.submit(self._download)
            return self.job

    def _download(self):
        process=None
        try:
            directory=self.config.resolve(self.config.value.model_directory)/'drama-3'
            log=self.root/'logs/model-download.log'
            log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('w') as stream:
                process=subprocess.Popen([sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/download_model.py'),
                    '--confirm-download','--destination',str(directory)],stdout=stream,stderr=subprocess.STDOUT)
                while process.poll() is None:
                    self.check()
                    completed=len(list(directory.glob('*.safetensors'))) if directory.exists() else 0
                    self.progress('downloading',f'Downloading Drama 3 · {completed}/8 weight files complete')
                    self.cancel_event.wait(0.5)
                if process.returncode:
                    raise RuntimeError('Model download failed. See logs/model-download.log for details.')
            with self.lock:
                self.state,self.message='unloaded','Drama 3 is available locally.'
        except Exception as exc:
            if process and process.poll() is None:
                process.terminate()
                try:process.wait(10)
                except subprocess.TimeoutExpired:process.kill(); process.wait()
            with self.lock:
                self.state,self.error='error',str(exc)

    def progress(self,stage,message):
        with self.lock:
            self.message=message
            self.stage=stage
        logger.info('%s: %s',stage,message)

    def check(self):
        if self.cancel_event.is_set():
            raise InterruptedError('Operation cancelled')
        if self.monitor:
            self.monitor.check()

    def _begin(self,state,allowed):
        # Caller holds the lifecycle lock until its worker has been submitted.
        if self.state not in allowed:
            raise RuntimeError(f'Cannot {state} while {self.state}')
        self.state,self.error,self.job=state,None,uuid.uuid4().hex
        self.cancel_event.clear()

    def load(self,path,vram_limit_gib=None):
        with self.lock:
            self._begin('loading',{'unloaded','error'})
            self.model_path=str(Path(path).expanduser().resolve())
            self.future=self.executor.submit(self._load,vram_limit_gib)
            return self.job

    def _load(self,limit):
        try:
            self.monitor=self.monitor_factory()
            self.monitor.__enter__()
            self.adapter=self.factory(limit,check=self.check,progress=self.progress)
            self.adapter.load(self.model_path)
            with self.lock:
                self.state,self.message='ready','Checkpoint loaded; speech quality remains unverified.'
        except Exception as exc:
            self._release()
            with self.lock:
                self.state,self.error='error',str(exc)
            logger.exception('Model loading failed')

    def generate(self,text,parameters):
        with self.lock:
            self._begin('generating',{'ready'})
            self.future=self.executor.submit(self._generate,self.job,text,parameters)
            return self.job

    def _generate(self,job,text,parameters):
        output=self.outputs/(job+'.wav')
        record=dict(id=job,text=text,created=time.time(),status='failed',parameters=dict(parameters))
        try:
            self.outputs.mkdir(parents=True,exist_ok=True)
            assignments=parameters.pop('voices',[])
            if assignments:
                parameters['references']=[self.voices.resolve(v['id'],v['speaker']) for v in assignments]
            metrics=self.adapter.generate(text,output,**parameters)
            self.check()
            record.update(status='audio_generated_unreviewed',metrics=metrics)
            self.progress('complete','Audio generated. Listen to verify intelligible speech.')
        except Exception as exc:
            record['error']=str(exc)
            with self.lock:
                self.error=str(exc)
            # Partial or safety-rejected output must never appear in successful history.
            try:
                output.unlink(missing_ok=True)
            except OSError:
                logger.exception('Could not clean rejected audio output')
            logger.exception('Generation failed')
        finally:
            record['codes_available']=output.with_suffix('.codes.npy').is_file()
            if self.monitor:
                record['monitor']=[s for s in self.monitor.samples if s['time']>=record['created']]
            try:
                metadata=output.with_suffix('.json')
                temp=metadata.with_suffix('.partial.json')
                temp.write_text(json.dumps(record,indent=2)+'\n')
                temp.replace(metadata)
            except OSError as exc:
                with self.lock:self.error=f'Audio history could not be written: {exc}'
                logger.exception('Could not persist generation metadata')
            finally:
                with self.lock:self.state='ready'

    def cancel(self):
        with self.lock:
            if self.state not in {'loading','generating','downloading'}:
                raise RuntimeError('No cancellable operation running')
            self.cancel_event.set()
            self.message='Cancellation requested; stopping at the next model operation boundary.'

    def unload(self):
        with self.lock:
            self._begin('unloading',{'ready','error','unloaded'})
            self.future=self.executor.submit(self._unload)
            return self.job

    def _release(self):
        if self.adapter:
            self.adapter.unload()
            self.unload_metrics=dict(gpu_allocated_after_unload_bytes=torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                                     gpu_reserved_after_unload_bytes=torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
                                     process_rss_after_unload_bytes=psutil.Process().memory_info().rss)
        self.adapter=None
        if self.monitor:
            self.monitor.__exit__(None,None,None)
        self.monitor=None

    def _unload(self):
        try:
            self._release()
            with self.lock:
                self.state,self.model_path,self.message='unloaded',None,'Model unloaded.'
        except Exception as exc:
            with self.lock:
                self.state,self.error='error',str(exc)

    def history(self):
        records=[]
        if self.outputs.exists():
            for path in self.outputs.glob('*.json'):
                if len(path.stem)!=32:
                    continue
                try:
                    # Parsed once per file version; the monitor samples are not sent to the UI.
                    key=(path,path.stat().st_mtime_ns)
                    if key not in self.record_cache:
                        record=json.loads(path.read_text())
                        record.pop('monitor',None)
                        self.record_cache[key]=record
                    record=self.record_cache[key]
                    if record['id']==path.stem:
                        wave=path.with_suffix('.wav')
                        if record.get('status')=='audio_generated_unreviewed' and wave.is_file():
                            metrics=record.setdefault('metrics',{})
                            if 'audio_warning' not in metrics or str(metrics.get('audio_warning','')).startswith('Nach etwa '):
                                key=(str(wave),wave.stat().st_mtime_ns)
                                if key not in self.activity_cache:
                                    try:
                                        info=sf.info(wave)
                                        if info.duration<=60 and info.channels==1:
                                            data,rate=sf.read(wave,dtype='float32')
                                            self.activity_cache[key]=audio_activity(data,rate)
                                    except (RuntimeError,ValueError,OSError):
                                        logger.warning('Could not inspect audio activity: %s',wave)
                                metrics.update(self.activity_cache.get(key,{}))
                        records.append(record)
                except (ValueError,KeyError,OSError):
                    logger.warning('Invalid generation metadata: %s',path)
        return sorted(records,key=lambda r:r['created'],reverse=True)

    def close(self):
        self.cancel_event.set()
        self.executor.shutdown(wait=True,cancel_futures=True)
        self._release()
