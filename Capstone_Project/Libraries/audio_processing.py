import librosa
import numpy as np

# Load the audio file
audio_path = 'path_to_your_audio_file.wav'
y, sr = librosa.load(audio_path, sr=None)  # sr (sample rate) is set to None to preserve the native sampling rate

# Extract MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Transpose the result to have the time dimension first
mfccs = np.transpose(mfccs, (1, 0))

# Normalize MFCCs
mfccs -= np.mean(mfccs, axis=0)
mfccs /= np.std(mfccs, axis=0)

# mfccs is now a 2D numpy array of shape (time_frames, mfcc_features), ready to be used as input to a neural network.

