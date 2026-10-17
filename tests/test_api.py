"""API lifecycle tests inject an explicit fake; they never claim model validation."""
import threading
from pathlib import Path

from fastapi.testclient import TestClient

from red_curtain.api import create_app
from red_curtain.manager import ModelManager


class NoGpuMonitor:
    samples=[]
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def check(self):pass


class BlockingAdapter:
    def __init__(self,*args,check,progress):
        self.check,self.progress=check,progress
        self.metrics={}
        self.entered=threading.Event()
        self.release=threading.Event()
    def load(self,path):
        if not Path(path).is_dir():raise ValueError('Missing model')
    def unload(self):pass
    def generate(self,text,output,**kwargs):
        self.entered.set()
        self.release.wait(5)
        self.check()
        raise ValueError('Deliberate decoder failure in test')


def manager(tmp_path):
    return ModelManager(tmp_path,adapter_factory=BlockingAdapter,monitor_factory=NoGpuMonitor)


def test_invalid_requests_and_foreign_origins_do_not_start_inference(tmp_path):
    m=manager(tmp_path)
    with TestClient(create_app(tmp_path,m),base_url='http://127.0.0.1:7860') as client:
        assert client.get('/api/status').json()['state']=='unloaded'
        assert client.post('/api/generate',json={'text':'Hello'}).status_code==409
        assert client.post('/api/generate',json={'text':'<|speaker:16|> Hello'}).status_code==422
        assert client.post('/api/model/load',json={'path':str(tmp_path),'vram_limit_gib':0}).status_code==422
        assert client.post('/api/model/load',headers={'Origin':'https://attacker.invalid'},
                           json={'path':str(tmp_path)}).status_code==403
        assert client.get('/api/audio/..%2Fsecret').status_code==404
        assert client.get('/api/audio/'+'a'*32).status_code==404
        assert client.get('/api/status',headers={'host':'attacker.invalid'}).status_code==400


def test_single_generation_cancel_failure_history_and_unload(tmp_path):
    m=manager(tmp_path)
    with TestClient(create_app(tmp_path,m),base_url='http://127.0.0.1:7860') as client:
        assert client.post('/api/model/load',json={'path':str(tmp_path)}).status_code==202
        m.future.result(5)
        assert m.status()['state']=='ready'
        assert client.post('/api/model/load',json={'path':str(tmp_path)}).status_code==409
        job=client.post('/api/generate',json={'text':'<|speaker:0|> [laughs] Hello'}).json()['job']
        assert m.adapter.entered.wait(5)
        assert client.post('/api/generate',json={'text':'second'}).status_code==409
        assert client.post('/api/model/unload').status_code==409
        assert client.post('/api/cancel').status_code==202
        m.adapter.release.set()
        m.future.result(5)
        record=client.get('/api/generations').json()[0]
        assert record['id']==job and record['status']=='failed'
        assert 'cancelled' in record['error']
        assert client.get('/api/audio/'+job).status_code==404
        assert not list(m.outputs.glob('*.wav'))
        assert client.post('/api/model/unload').status_code==202
        m.future.result(5)
        assert m.status()['state']=='unloaded'


def test_load_failure_is_visible_and_recoverable(tmp_path):
    m=manager(tmp_path)
    with TestClient(create_app(tmp_path,m),base_url='http://127.0.0.1:7860') as client:
        client.post('/api/model/load',json={'path':str(tmp_path/'missing')})
        m.future.result(5)
        assert m.status()['state']=='error'
        assert 'Missing model' in m.status()['error']
        client.post('/api/model/load',json={'path':str(tmp_path)})
        m.future.result(5)
        assert m.status()['state']=='ready'


def test_audio_route_only_serves_successful_internal_history_ids(tmp_path):
    import json
    import numpy as np
    from red_curtain.audio import write_wav
    m=manager(tmp_path)
    m.outputs.mkdir()
    identifier='a'*32
    output=m.outputs/(identifier+'.wav')
    write_wav(output,np.sin(np.arange(4410)*0.1)*0.2,44100)
    record=dict(id=identifier,status='audio_generated_unreviewed',created=1,text='unit test signal')
    output.with_suffix('.json').write_text(json.dumps(record))
    with TestClient(create_app(tmp_path,m),base_url='http://127.0.0.1:7860') as client:
        response=client.get('/api/audio/'+identifier)
        assert response.status_code==200 and response.content[:4]==b'RIFF'
        assert response.headers['content-type']=='audio/wav'
        record['status']='failed'
        output.with_suffix('.json').write_text(json.dumps(record))
        assert client.get('/api/audio/'+identifier).status_code==404


def test_settings_persist_and_download_requires_explicit_confirmation(tmp_path):
    m=manager(tmp_path)
    with TestClient(create_app(tmp_path,m),base_url='http://127.0.0.1:7860') as client:
        assert client.post('/api/models/download',json={}).status_code==422
        value=dict(model_directory='custom-models',output_directory='custom-audio',vram_limit_gib=11.5)
        assert client.put('/api/settings',json=value).status_code==200
        assert client.get('/api/settings').json()==value
        assert m.outputs==tmp_path/'custom-audio'
        client.post('/api/model/load',json={'path':str(tmp_path)})
        m.future.result(5)
        assert client.put('/api/settings',json=value).status_code==409
    second=manager(tmp_path)
    assert second.config.value.vram_limit_gib==11.5
    second.close()


def test_reference_upload_ignores_filename_and_rejects_invalid_audio(tmp_path):
    import io
    import numpy as np
    import soundfile as sf
    buffer=io.BytesIO()
    sf.write(buffer,np.sin(np.arange(44100)*0.03)*0.1,44100,format='WAV')
    m=manager(tmp_path)
    with TestClient(create_app(tmp_path,m),base_url='http://127.0.0.1:7860') as client:
        result=client.post('/api/voices',data={'name':'Test voice','text':'Reference transcript'},
            files={'audio':('../../unsafe.wav',buffer.getvalue(),'audio/wav')})
        assert result.status_code==201,result.text
        voice=result.json()
        assert len(voice['id'])==32 and voice['name']=='Test voice'
        assert (tmp_path/'data/voices'/f"{voice['id']}.wav").exists()
        assert not (tmp_path/'unsafe.wav').exists()
        assert client.get('/api/voices').json()[0]['id']==voice['id']
        invalid=client.post('/api/voices',data={'name':'Bad','text':'Transcript'},
            files={'audio':('bad.wav',b'not audio','audio/wav')})
        assert invalid.status_code==422
        assert not list((tmp_path/'data/uploads').glob('*'))


def test_output_filesystem_failure_does_not_leave_manager_busy(tmp_path):
    m=manager(tmp_path)
    with TestClient(create_app(tmp_path,m),base_url='http://127.0.0.1:7860') as client:
        client.post('/api/model/load',json={'path':str(tmp_path)})
        m.future.result(5)
        m.outputs.write_text('A file replaced the intended output directory')
        assert client.post('/api/generate',json={'text':'Hello'}).status_code==202
        m.future.result(5)
        assert m.status()['state']=='ready'
        assert m.status()['error']
        assert client.post('/api/model/unload').status_code==202
        m.future.result(5)
        assert m.status()['state']=='unloaded'


def test_decoder_failure_preserves_codes_but_never_serves_failed_audio(tmp_path):
    import numpy as np
    class DecoderFailure(BlockingAdapter):
        def generate(self,text,output,**kwargs):
            np.save(output.with_suffix('.codes.npy'),np.zeros((10,2),dtype=np.int64))
            raise RuntimeError('Decoder workspace exhausted in test')
    m=ModelManager(tmp_path,adapter_factory=DecoderFailure,monitor_factory=NoGpuMonitor)
    with TestClient(create_app(tmp_path,m),base_url='http://127.0.0.1:7860') as client:
        client.post('/api/model/load',json={'path':str(tmp_path)})
        m.future.result(5)
        job=client.post('/api/generate',json={'text':'Test'}).json()['job']
        m.future.result(5)
        assert (m.outputs/(job+'.codes.npy')).is_file()
        record=client.get('/api/generations').json()[0]
        assert record['status']=='failed' and record['codes_available']
        assert client.get('/api/audio/'+job).status_code==404


def test_old_history_gets_silence_warning_without_modifying_original_metadata(tmp_path):
    import json
    import numpy as np
    import soundfile as sf
    m=manager(tmp_path)
    m.outputs.mkdir()
    identifier='e'*32
    data=np.concatenate([np.sin(np.arange(32000)*.1)*.15,np.zeros(16000*8)])
    sf.write(m.outputs/(identifier+'.wav'),data,16000)
    metadata=m.outputs/(identifier+'.json')
    original=json.dumps(dict(id=identifier,status='audio_generated_unreviewed',created=1,text='Example text',metrics={}))
    metadata.write_text(original)
    with TestClient(create_app(tmp_path,m),base_url='http://127.0.0.1:7860') as client:
        result=client.get('/api/generations').json()[0]
        assert result['metrics']['audio_warning']
        assert result['metrics']['trailing_silence_s']>=7.9
        assert metadata.read_text()==original
