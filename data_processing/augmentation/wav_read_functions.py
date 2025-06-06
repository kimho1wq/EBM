import random
import numpy as np
import soundfile as sf

def rand_crop_audio(audio, size: int, get_start_time=False, samplerate=16000):
    '''Read '~.wav' file and crop it. 
    If wav is short, pad wav with zeros.
    '''
    num_frame = len(audio)
    
    # padding
    if len(audio) <= size:
        if audio.shape[0] < size:
            crop_audio = _pad_wav(audio, size)
        elif size <= audio.shape[0]:
            crop_audio = audio[:size]
        start = 0

    # random crop
    else: 
        while(True):
            try:
                start = random.randint(0, num_frame - size) 
                crop_audio = audio[start: start + size]
                break
            except:
                pass
            
    if get_start_time:
        return crop_audio, start / samplerate
    else:
        return crop_audio

def rand_crop_read(path: str, size: int, get_start_time=False):
    '''Read '~.wav' file and crop it. 
    If wav is short, pad wav with zeros.
    '''
    sf_info = sf.info(path)
    num_frame = int((sf_info.duration - 0.001) * sf_info.samplerate)  

    # padding
    if num_frame <= size:
        wav, _ = sf.read(path)
        if wav.shape[0] < size:
            wav = _pad_wav(wav, size)
        elif size <= wav.shape[0]:
            wav = wav[:size]
        start = 0

    # random crop
    else: 
        while(True):
            try:
                start = random.randint(0, num_frame - size) 
                wav, _ = sf.read(path, start=start, stop=start + size)
                break
            except:
                with open('/code/error.txt', 'a') as f:
                    f.write(f'{path}\n')
            
    if get_start_time:
        return wav, start / sf_info.samplerate
    else:
        return wav

def linspace_crop_read(path: str, num_seg: int, seg_size: int, get_org=False):
    '''Read '~.wav' file and divide it into several segments using linspace function.
    '''
    wav, _ = sf.read(path)
    
    if wav.shape[0] < seg_size:
        wav = _pad_wav(wav)
        
    buffer = []
    indices = np.linspace(0, wav.shape[0] - seg_size, num_seg)
    for idx in indices:
        idx = int(idx)
        buffer.append(wav[idx:idx + seg_size])
    buffer = np.stack(buffer, axis=0)
    
    if get_org:
        return buffer, wav
    else:
        return buffer

def _pad_wav(wav, size):
    if len(np.shape(wav)) == 1:
        shortage = size - wav.shape[0]
        wav = np.pad(wav, (0, shortage), 'wrap')
        return wav
    elif len(np.shape(wav)) == 2:
        shortage = size - wav.shape[0]
        wav = np.pad(wav, ((0, shortage), (0, 0)), 'wrap')
        return wav
    else:
        raise Exception()