"""Local backend API. Full studio UI is gated on real inference validation."""
from contextlib import asynccontextmanager
import json
from pathlib import Path
import re
import logging
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from red_curtain.manager import ModelManager
from red_curtain.config import Settings
import uuid


class LoadRequest(BaseModel):
    path:str=Field(min_length=1,max_length=4096)
    vram_limit_gib:float|None=Field(default=None,ge=1,allow_inf_nan=False)


class VoiceAssignment(BaseModel):
    id:str=Field(pattern='^[0-9a-f]{32}$')
    speaker:int=Field(ge=0,le=15)


class GenerateRequest(BaseModel):
    text:str=Field(min_length=1,max_length=10000)
    max_new_tokens:int|None=Field(default=None,ge=1,le=1024)  # None: limit from the text length
    temperature:float=Field(default=1.0,gt=0,le=2)
    top_p:float=Field(default=0.9,gt=0,le=1)
    top_k:int=Field(default=0,ge=0,le=4096)  # 0: off
    seed:int=Field(default=42,ge=0,le=2**32-1)
    part_size:int=Field(default=300,ge=100,le=300)
    context:bool=True
    voices:list[VoiceAssignment]=Field(default_factory=list,max_length=4)


def create_app(root=None,manager=None):
    root=Path(root or Path(__file__).resolve().parents[1])
    manager=manager or ModelManager(root)
    @asynccontextmanager
    async def lifespan(app):
        logs=root/'logs'
        logs.mkdir(parents=True,exist_ok=True)
        handler=RotatingFileHandler(logs/'red-curtain.log',maxBytes=5*1024**2,backupCount=3)
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
        logger=logging.getLogger('red_curtain')
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        try:
            yield
        finally:
            import asyncio
            await asyncio.to_thread(manager.close)
            logger.removeHandler(handler)
            handler.close()
    # No external Swagger CDN or telemetry during local runtime.
    app=FastAPI(title='Red Curtain backend',docs_url=None,redoc_url=None,lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=['127.0.0.1','localhost'])

    @app.middleware('http')
    async def local_origin(request:Request,call_next):
        origin=request.headers.get('origin')
        if request.method not in {'GET','HEAD','OPTIONS'} and origin and origin not in {
            'http://127.0.0.1:7860','http://localhost:7860'}:
            from fastapi.responses import JSONResponse
            return JSONResponse({'detail':'Only same-origin local requests are accepted'},status_code=403)
        return await call_next(request)

    @app.get('/api/info')
    def root_status():
        return dict(application='Red Curtain',runtime='local ROCm',status='/api/status')

    @app.get('/api/status')
    def status():
        return manager.status()

    @app.get('/api/models')
    def models():
        found=[]
        directory=manager.config.resolve(manager.config.value.model_directory)
        if directory.exists():
            for path in sorted(directory.iterdir()):
                config=path/'config.json'
                if not path.is_dir() or not config.is_file():
                    continue
                try:
                    raw=json.loads(config.read_text())
                    if raw.get('model_type')!='fish_omni':
                        continue
                    found.append(dict(name=path.name,path=str(path.resolve()),model_type=raw.get('model_type'),
                        quantization=raw.get('quantization_config',{}).get('quant_method'),
                        compatibility=manager.compatibility(path)))
                except (OSError,ValueError):
                    continue
        return found

    def operation(function,*args):
        try:
            return {'job':function(*args)}
        except (ValueError,RuntimeError) as exc:
            raise HTTPException(409,str(exc)) from exc

    @app.get('/api/settings')
    def settings():
        return manager.config.value.model_dump()

    @app.put('/api/settings')
    def save_settings(body:Settings):
        operation(manager.save_settings,body)
        return manager.config.value.model_dump()

    @app.get('/api/logs')
    def logs():
        path=root/'logs/red-curtain.log'
        if not path.exists():return {'text':''}
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-24000))
            return {'text':stream.read().decode('utf-8',errors='replace')}

    @app.post('/api/models/download',status_code=202)
    def download(body:dict):
        if body.get('confirm_download') is not True:
            raise HTTPException(422,'Explicit download confirmation is required')
        return operation(manager.download)

    @app.get('/api/voices')
    def voices():
        return manager.voices.list()

    @app.post('/api/voices',status_code=201)
    async def add_voice(name:str=Form(...),text:str=Form(...),audio:UploadFile=File(...)):
        directory=root/'data/uploads'
        directory.mkdir(parents=True,exist_ok=True)
        temporary=directory/(uuid.uuid4().hex+'.upload')
        size=0
        try:
            with temporary.open('wb') as stream:
                while chunk:=await audio.read(1024**2):
                    size+=len(chunk)
                    if size>20*1024**2:raise HTTPException(413,'Reference exceeds 20 MiB')
                    stream.write(chunk)
            import asyncio
            try:return await asyncio.to_thread(manager.voices.add,temporary,name,text)
            except (ValueError,RuntimeError) as exc:raise HTTPException(422,str(exc)) from exc
        finally:
            temporary.unlink(missing_ok=True)
            await audio.close()

    @app.post('/api/model/load',status_code=202)
    def load(body:LoadRequest):
        return operation(manager.load,body.path,body.vram_limit_gib)

    @app.post('/api/model/unload',status_code=202)
    def unload():
        return operation(manager.unload)

    @app.post('/api/generate',status_code=202)
    def generate(body:GenerateRequest):
        from red_curtain.fish_omni.generate import validate_drama3_text
        try:
            validate_drama3_text(body.text,'text')
        except ValueError as exc:
            raise HTTPException(422,str(exc)) from exc
        return operation(manager.generate,body.text,body.model_dump(exclude={'text'}))

    @app.post('/api/cancel',status_code=202)
    def cancel():
        operation(manager.cancel)
        return {'message':'Cancellation requested'}

    @app.get('/api/generations')
    def history():
        return manager.history()

    @app.get('/api/audio/{generation_id}')
    def audio(generation_id:str):
        if not re.fullmatch('[0-9a-f]{32}',generation_id):
            raise HTTPException(404,'Unknown generation')
        record=next((r for r in manager.history() if r['id']==generation_id),None)
        path=manager.outputs/(generation_id+'.wav')
        if not record or record.get('status')!='audio_generated_unreviewed' or not path.is_file() or path.is_symlink():
            raise HTTPException(404,'No successful audio generation')
        return FileResponse(path,media_type='audio/wav',filename=path.name)

    dist=Path(__file__).resolve().parents[1]/'frontend/dist'
    if (dist/'index.html').exists():
        app.mount('/assets',StaticFiles(directory=dist/'assets'),name='assets')
        @app.get('/')
        def studio():return FileResponse(dist/'index.html')
    else:
        @app.get('/')
        def missing_frontend():return {'message':'Build the local UI with: cd frontend && npm ci && npm run build'}
    return app
