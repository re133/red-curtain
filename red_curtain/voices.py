"""Local reference library; filenames are generated IDs, never uploaded names."""
import json
from pathlib import Path
import re
import time
import uuid
import soundfile as sf

from red_curtain.audio import read_reference
from red_curtain.fish_omni.generate import validate_drama3_text


class VoiceLibrary:
    def __init__(self,root):
        self.directory=Path(root)/'data/voices'

    def list(self):
        result=[]
        if self.directory.exists():
            for path in self.directory.glob('*.json'):
                try:
                    item=json.loads(path.read_text())
                    if item['id']==path.stem and (self.directory/(item['id']+'.wav')).is_file():
                        result.append(item)
                except (ValueError,KeyError,OSError):
                    continue
        return sorted(result,key=lambda r:r['created'],reverse=True)

    def add(self,source,name,text):
        if not name.strip() or len(name)>80 or not 1<=len(text)<=2000:
            raise ValueError('Use a voice name up to 80 characters and a transcript up to 2000 characters')
        validate_drama3_text(text,'reference transcript')
        if '<|speaker:' in text:
            raise ValueError('Choose the reference speaker separately; omit speaker tags in the transcript')
        audio=read_reference(source)
        identifier=uuid.uuid4().hex
        self.directory.mkdir(parents=True,exist_ok=True)
        target=self.directory/(identifier+'.wav')
        sf.write(target,audio[0,0].numpy(),44100,subtype='PCM_16')
        record=dict(id=identifier,name=name.strip(),text=text,duration_s=audio.shape[-1]/44100,created=time.time())
        try:
            target.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return record

    def resolve(self,identifier,speaker):
        if not re.fullmatch('[0-9a-f]{32}',identifier):
            raise ValueError('Invalid voice ID')
        record=next((r for r in self.list() if r['id']==identifier),None)
        if record is None:
            raise ValueError('Unknown saved voice')
        return dict(path=str(self.directory/(identifier+'.wav')),text=record['text'],speaker=speaker)
