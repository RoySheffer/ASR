# Data augmentation is the process by which we create new polymerized data samples by adding small disturbance on our initial training set.
# To generate polymerized data for audio, we can apply noise injection, shifting time, changing pitch and speed.
# The objective is to make our model invariant to those disturbance and enhace its ability to generalize.
# In order to this to work adding the disturbance must conserve the same label as the original training sample.

import numpy as np
import librosa
import torchaudio

# NOISE
def noise(data, noise_amp=0.035):
    noise_amp = noise_amp * np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

# STRETCH
def stretch(data, time_strech_amp=0.1):
    time_strech_rate = 1.0 + np.clip(time_strech_amp * np.random.uniform(), -0.5, 0.5)
    return librosa.effects.time_stretch(data, rate=time_strech_rate)

# PITCH
def pitch(data, sampling_rate, pitch_amp=5):
    n_steps = np.clip(pitch_amp * np.random.uniform(), -15, 15)
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=n_steps)

def augment(data, sample_rate):
    noise_amp = 0.005
    pitch_amp = 3
    time_strech_amp = 0.1
    # data = noise(data, noise_amp=noise_amp)
    data = stretch(data, time_strech_amp=time_strech_amp)
    data = pitch(data, sample_rate, pitch_amp=pitch_amp)
    return data


