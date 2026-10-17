import numpy as np
import pytest
import soundfile as sf

from red_curtain.audio import write_wav, read_reference, audio_activity


@pytest.mark.parametrize('samples',[
    [],[float('nan')],[float('inf')],[0,0],[2.0,-2.0],[[0.1,0.2]],
])
def test_invalid_audio_never_becomes_output(tmp_path,samples):
    path=tmp_path/'output.wav'
    with pytest.raises(ValueError):
        write_wav(path,samples,44100)
    assert not path.exists()
    assert not list(tmp_path.glob('*.partial.wav'))


def test_pcm_roundtrip_preserves_length_rate_and_signal(tmp_path):
    # A deterministic test signal, explicitly not model inference.
    samples=np.sin(np.arange(4410)*2*np.pi*440/44100).astype(np.float32)*0.2
    path=tmp_path/'nested'/'unit-test-tone.wav'
    info=write_wav(path,samples,44100)
    restored,rate=sf.read(path)
    assert rate==44100 and len(restored)==len(samples)
    assert info['duration_s']==0.1
    np.testing.assert_allclose(restored,samples,atol=1/32768)


def test_local_reference_resampling_and_duration_guard(tmp_path):
    rate=48000
    samples=np.sin(np.arange(rate)*2*np.pi*220/rate)*0.1
    path=tmp_path/'reference.wav'
    sf.write(path,np.stack([samples,samples],axis=1),rate)
    tensor=read_reference(path)
    assert tuple(tensor.shape)==(1,1,44100)
    assert tensor.isfinite().all()
    sf.write(path,np.zeros(100),rate)
    with pytest.raises(ValueError,match='between'):
        read_reference(path)
    sf.write(path,np.zeros(rate),rate)
    with pytest.raises(ValueError,match='silence'):
        read_reference(path)


def test_long_near_silent_tail_is_warned_about_without_cutting_audio(tmp_path):
    rate=16000
    speech=np.sin(np.arange(rate*2)*0.1)*0.15
    noise=np.sin(np.arange(rate*18)*0.37)*0.0005
    samples=np.concatenate((speech,noise))
    path=tmp_path/'trailing-silence.wav'
    info=write_wav(path,samples,rate)
    assert info['audio_warning']
    assert info['energy_active_end_s']==pytest.approx(2,abs=.1)
    assert info['trailing_silence_s']==pytest.approx(18,abs=.1)
    assert sf.info(path).frames==len(samples)


def test_ordinary_pause_and_quiet_audio_do_not_get_a_long_tail_warning():
    rate=16000
    quiet=np.sin(np.arange(rate*6)*.1)*0.001
    quiet[rate*2:rate*3]=0
    assert not audio_activity(quiet,rate)['audio_warning']
