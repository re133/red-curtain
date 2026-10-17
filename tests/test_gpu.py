"""Read-only telemetry fixtures, including remapped multi-GPU device indices."""
import json
import pytest

from red_curtain import gpu
from red_curtain.manager import ModelManager
from red_curtain.safety import check_snapshot


def test_sysfs_uses_selected_pci_address_and_available_sensors_without_writes(tmp_path):
    selected=tmp_path/'0000:03:00.0'
    other=tmp_path/'0000:04:00.0'
    for directory,temp in [(selected,'43000'),(other,'120000')]:
        monitor=directory/'hwmon/hwmon7'
        monitor.mkdir(parents=True)
        (monitor/'temp1_input').write_text(temp)
        (monitor/'temp1_label').write_text('edge')
        (monitor/'temp1_crit').write_text('95000')
        (directory/'mem_info_vram_total').write_text(str(24*1024**3))
        (directory/'mem_info_vram_used').write_text(str(3*1024**3))
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    snapshot=gpu.sysfs_snapshot({'pci_bus_id':'0000:03:00.0','name':'Fixture GPU'},tmp_path)
    row=snapshot['selected']
    assert row['Temperature (Sensor edge) (C)']=='43.0'
    assert row['Temperature critical (Sensor edge) (C)']=='95.0'
    assert 'Temperature (Sensor memory) (C)' not in row
    assert row['Card Series']=='Fixture GPU'
    check_snapshot(snapshot)
    assert before=={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_rocm_smi_maps_logical_device_by_pci_instead_of_card_number():
    raw={'card0':{'PCI Bus':'0000:04:00.0'},'card1':{'PCI Bus':'00000000:03:00.0'}}
    assert gpu.select_smi_device(raw,{'pci_bus_id':'0000:03:00.0'})=={'card1':raw['card1']}
    with pytest.raises(RuntimeError,match='Cannot match'):
        gpu.select_smi_device(raw,{'pci_bus_id':'0000:05:00.0'})


def test_telemetry_fallback_and_optional_sensor_notice(monkeypatch):
    monkeypatch.setattr(gpu,'gpu_identity',lambda _:dict(available=True,pci_bus_id='0000:03:00.0'))
    monkeypatch.setattr(gpu,'sysfs_snapshot',lambda _:None)
    monkeypatch.setattr(gpu,'smi_snapshot',lambda _:{'card1':{'Temperature (Sensor edge) (C)':'42'}})
    result=gpu.read_gpu_snapshot()
    check_snapshot(result)
    assert 'junction, memory' in result['card1']['Monitoring notice']


def test_missing_telemetry_fails_closed(monkeypatch):
    monkeypatch.setattr(gpu,'gpu_identity',lambda _:dict(available=True,pci_bus_id='0000:03:00.0'))
    monkeypatch.setattr(gpu,'sysfs_snapshot',lambda _:None)
    def unavailable(_):raise FileNotFoundError('No rocm-smi')
    monkeypatch.setattr(gpu,'smi_snapshot',unavailable)
    with pytest.raises(RuntimeError,match='telemetry is unavailable'):
        gpu.read_gpu_snapshot()


def test_driver_memory_accounting_can_reduce_overoptimistic_hip_free_memory(monkeypatch):
    gib=1024**3
    monkeypatch.setattr(gpu,'gpu_identity',lambda _:dict(available=True))
    monkeypatch.setattr(gpu.torch.cuda,'mem_get_info',lambda _:(16*gib,16*gib))
    monkeypatch.setattr(gpu.torch.cuda,'memory_reserved',lambda _:3*gib)
    monkeypatch.setattr(gpu,'sysfs_snapshot',lambda _:{'selected':{
        'VRAM Total Memory (B)':str(16*gib),'VRAM Total Used Memory (B)':str(6*gib)}})
    assert gpu.device_memory()==dict(total_bytes=16*gib,free_bytes=10*gib,reserved_bytes=3*gib)


def test_validation_badge_requires_matching_gpu_and_checkpoint(tmp_path,monkeypatch):
    import red_curtain.manager as module
    model=tmp_path/'model'
    model.mkdir()
    (model/'config.json').write_text('{}')
    (model/'download-provenance.json').write_text(json.dumps({'revision':'abc'}))
    evidence={'model_revision':'abc','gpu':'Validated GPU','architecture':'gfx-test'}
    (tmp_path/'tested.json').write_text(json.dumps(evidence))
    manager=ModelManager(tmp_path)
    manager.tested=tmp_path/'tested.json'
    try:
        monkeypatch.setattr(module,'gpu_identity',lambda _:dict(available=True,name='Other GPU',architecture='gfx-other'))
        assert manager.compatibility(model)=='experimental_unverified'
        monkeypatch.setattr(module,'gpu_identity',lambda _:dict(available=True,name='Validated GPU',architecture='gfx-test'))
        assert manager.compatibility(model)=='tested_rocm'
        (model/'download-provenance.json').write_text(json.dumps({'revision':'different'}))
        assert manager.compatibility(model)=='experimental_unverified'
    finally:manager.close()
