"""GPU identity and read-only Linux telemetry for the HIP device used by the adapter."""
import ctypes
import ctypes.util
from functools import lru_cache
import json
from pathlib import Path
import re
import subprocess

import torch

PCI_ROOT = Path('/sys/bus/pci/devices')


def normalize_pci(value):
    match = re.fullmatch(r'([0-9a-f]{4,8}):([0-9a-f]{2}):([0-9a-f]{2})\.([0-7])', str(value).strip().lower())
    if match:
        domain, bus, slot, function = match.groups()
        return f'{int(domain,16):04x}:{bus}:{slot}.{function}'
    return None


def _hip_pci(device):
    candidates = [ctypes.util.find_library('amdhip64'), 'libamdhip64.so',
                  str(Path(torch.__file__).parent/'lib/libamdhip64.so')]
    for name in filter(None, candidates):
        try:
            library = ctypes.CDLL(name)
            function = library.hipDeviceGetPCIBusId
            function.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
            function.restype = ctypes.c_int
            buffer = ctypes.create_string_buffer(32)
            if function(buffer, len(buffer), device) == 0:
                return normalize_pci(buffer.value.decode('ascii'))
        except (OSError, AttributeError):
            continue
    return None


@lru_cache(maxsize=8)
def gpu_identity(device=0):
    if not torch.version.hip or not torch.cuda.is_available():
        return dict(available=False, index=device, name=None, architecture=None,
                    pci_bus_id=None, total_vram_bytes=0)
    properties = torch.cuda.get_device_properties(device)
    # HIP gives the mapping after visibility filters, unlike DRM card numbering.
    pci = _hip_pci(device)
    if pci is None and all(hasattr(properties, k) for k in ('pci_bus_id','pci_device_id','pci_domain_id')):
        pci = f'{properties.pci_domain_id:04x}:{properties.pci_bus_id:02x}:{properties.pci_device_id:02x}.0'
    return dict(available=True, index=device, name=properties.name,
                architecture=getattr(properties, 'gcnArchName', '').split(':')[0],
                pci_bus_id=pci, total_vram_bytes=properties.total_memory)


def _read(path):
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _integer(path):
    try:
        return int(_read(path))
    except (ValueError, TypeError):
        return None


def sysfs_snapshot(identity, pci_root=PCI_ROOT):
    pci = identity.get('pci_bus_id')
    if not pci:
        return None
    directory = Path(pci_root)/pci
    row = {'Card Series': identity.get('name') or 'AMD GPU', 'PCI Bus': pci,
           'Telemetry source': 'Linux amdgpu hwmon (read-only)'}
    for key, filename in [('VRAM Total Memory (B)','mem_info_vram_total'),
                          ('VRAM Total Used Memory (B)','mem_info_vram_used'),
                          ('GPU use (%)','gpu_busy_percent')]:
        value = _integer(directory/filename)
        if value is not None:
            row[key] = str(value)
    found = False
    for monitor in sorted(directory.glob('hwmon/hwmon*')):
        for path in sorted(monitor.glob('temp*_input')):
            stem = path.name.removesuffix('_input')
            label = (_read(monitor/(stem+'_label')) or stem).lower()
            label = {'mem':'memory', 'vram':'memory', 'hotspot':'junction', 'gpu':'edge'}.get(label,label)
            raw = _read(path)
            if raw is None:
                continue
            found = True
            try:
                value = str(float(raw)/1000)
            except ValueError:
                value = raw  # Invalid readings are rejected by the safety policy.
            row[f'Temperature (Sensor {label}) (C)'] = value
            critical = _integer(monitor/(stem+'_crit'))
            if critical is not None and critical > 0:
                row[f'Temperature critical (Sensor {label}) (C)'] = str(critical/1000)
    return {'selected': row} if found else None


def select_smi_device(raw, identity):
    pci = identity.get('pci_bus_id')
    cards = {k:v for k,v in raw.items() if isinstance(v,dict)}
    matches = {k:v for k,v in cards.items() if pci and normalize_pci(v.get('PCI Bus')) == pci}
    if len(matches) == 1:
        return matches
    if len(cards) == 1 and not next(iter(cards.values())).get('PCI Bus'):
        return cards
    raise RuntimeError('Cannot match ROCm temperature telemetry to the selected GPU PCI address')


def smi_snapshot(identity):
    result = subprocess.run(['rocm-smi','--showbus','--showproductname','--showtemp',
                             '--showmeminfo','vram','--showuse','--json'],
                            capture_output=True,text=True,timeout=10,check=True)
    raw = json.loads(result.stdout[result.stdout.index('{'):])
    selected = select_smi_device(raw,identity)
    for row in selected.values():
        row.setdefault('Card Series',identity.get('name') or 'AMD GPU')
        row['Telemetry source'] = 'rocm-smi'
    return selected


def read_gpu_snapshot(device=0):
    identity = gpu_identity(device)
    if not identity['available']:
        raise RuntimeError('No ROCm GPU is available for temperature monitoring')
    snapshot = sysfs_snapshot(identity)
    if snapshot is None:
        try:
            snapshot = smi_snapshot(identity)
        except (OSError,ValueError,subprocess.SubprocessError) as exc:
            raise RuntimeError('GPU temperature telemetry is unavailable: expose read-only amdgpu hwmon '
                               'or provide rocm-smi for the selected GPU') from exc
    for row in snapshot.values():
        available = [k[len('Temperature (Sensor '):-len(') (C)')] for k,v in row.items()
                     if k.startswith('Temperature (Sensor ') and str(v) not in {'N/A','None',''}]
        missing = [label for label in ('edge','junction','memory') if label not in available]
        row['Monitoring notice'] = (
            f"Monitoring available temperature sensors: {', '.join(available) or 'none'}. "
            f"Not exposed by this device/driver: {', '.join(missing)}." if missing else '')
    return snapshot


def device_memory(device=0):
    """Use the more conservative of HIP free memory and driver VRAM accounting."""
    identity = gpu_identity(device)
    if not identity['available']:
        raise RuntimeError('A working ROCm GPU is required')
    free, total = torch.cuda.mem_get_info(device)
    snapshot = sysfs_snapshot(identity)
    if snapshot is None or 'VRAM Total Used Memory (B)' not in next(iter(snapshot.values())):
        try:
            snapshot = smi_snapshot(identity)
        except (OSError,ValueError,RuntimeError,subprocess.SubprocessError):
            snapshot = None
    if snapshot:
        row = next(iter(snapshot.values()))
        try:
            physical_free = max(0,int(row['VRAM Total Memory (B)'])-int(row['VRAM Total Used Memory (B)']))
            free = min(free,physical_free)
        except (KeyError,TypeError,ValueError):
            pass
    return dict(total_bytes=total,free_bytes=free,reserved_bytes=torch.cuda.memory_reserved(device))
