"""Project-local persisted settings; changes are atomic and take effect between jobs."""
import json
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict


class Settings(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    model_directory:str=Field(default='models',min_length=1,max_length=4096)
    output_directory:str=Field(default='outputs',min_length=1,max_length=4096)
    vram_limit_gib:float|None=Field(default=None,ge=1)


class ConfigStore:
    def __init__(self,root):
        self.root=Path(root).resolve()
        self.path=self.root/'data/settings.json'
        self.value=Settings.model_validate_json(self.path.read_text()) if self.path.exists() else Settings()

    def resolve(self,path):
        result=Path(path).expanduser()
        return (self.root/result).resolve() if not result.is_absolute() else result.resolve()

    def save(self,value):
        for directory in (value.model_directory,value.output_directory):
            path=self.resolve(directory)
            if path.exists() and not path.is_dir():
                raise ValueError(f'Not a directory: {path}')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temporary=self.path.with_suffix('.partial.json')
        temporary.write_text(value.model_dump_json(indent=2)+'\n')
        temporary.replace(self.path)
        self.value=value
