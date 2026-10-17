#!/usr/bin/env python3
"""Read at most 1 MiB of pinned codec metadata, never the full weight archive."""
from collections import OrderedDict
import io
import json
from pathlib import Path
import pickle
import struct
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
REVISION='e0a9218aa48888270f06ef1ed9ea67622c0e17d2'


class MetadataUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name)==('collections','OrderedDict'):
            return OrderedDict
        if module=='torch' and name.endswith('Storage'):
            return name
        if module=='torch._utils' and name in {'_rebuild_tensor_v2','_rebuild_tensor'}:
            return lambda storage,offset,size,stride,*args: dict(storage=storage,offset=offset,shape=size,stride=stride)
        raise ValueError(f'Forbidden pickle global: {module}.{name}')

    def persistent_load(self,identifier):
        return identifier


def main():
    url=f'https://huggingface.co/fishaudio/drama-3/resolve/{REVISION}/codec.pth'
    request=urllib.request.Request(url,headers={'Range':'bytes=0-1048575'})
    with urllib.request.urlopen(request,timeout=30) as response:
        if response.status!=206:
            raise RuntimeError('Server ignored bounded range; download cancelled')
        data=response.read(1048576)
    if data[:4]!=b'PK\x03\x04':
        raise ValueError('Expected a torch ZIP archive')
    name_length,extra_length=struct.unpack('<HH',data[26:30])
    name=data[30:30+name_length]
    if not name.endswith(b'/data.pkl'):
        raise ValueError(f'Unexpected first archive entry: {name!r}')
    metadata=MetadataUnpickler(io.BytesIO(data[30+name_length+extra_length:])).load()
    path=ROOT/'tests/fixtures/codec-tensor-metadata.json'
    path.write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'{len(metadata)} tensor metadata records; no tensor storage downloaded intentionally: {path}')


if __name__=='__main__':
    main()
