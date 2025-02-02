"""
Defines the features for pairwise distance measures.

It either does this by avaraging the MFCC features over time or by calculating
the minimal dynamic time warping (dtw) distance.

This code is loosely informed by

Bartelds , M., C. Richter , M. Liberman , and M. Wieling : A new
acoustic-based pronunciation distance measure. Frontiers in Artificial
Intelligence, 3, p. 39, 2020

"""


import librosa
import librosa.feature
import numpy as np


# NOTE: NOT used at the moment; look at mfcc_edd
def mfcc(sig, sr):
    """
    calculates mfcc features for this package.

    This function therefore, defines the specific way of calculating the mfcc
    features. To calculate it, this function uses librosa.

    The funtion resamples te signal to a 16000 Hz signal and extracts 12 mfcc
    coefficients.

    Returns
    -------
    features : (sequence, channels)

    """
    target_sr = 16000
    if sr != target_sr:
        sig = librosa.resample(sig, orig_sr=sr, target_sr=target_sr)

    # shape = (channel, time steps)
    mfcc_ = librosa.feature.mfcc(y=sig, sr=target_sr, n_mfcc=12, dct_type=2, norm='ortho')

    mfcc_ = mfcc_.T
    return mfcc_


def mfcc_edd(sig, sr):
    """
    calculates the mfccs, delta mfccs, delta delta mfccs, the energy (rms),
    and concatenates them.

    Uses librosa for all calculations. Resamples signal to 16000 Hz.

    Returns
    -------
    features : (sequence, channels)

    """
    target_sr = 16000
    # window size 25 ms
    if target_sr == 16000:
        window_size = 400
    else:
        raise ValueError("specify window_size")

    # step size 10 ms
    if target_sr == 16000:
        step_size = 160
    else:
        raise ValueError("specify step_size")

    if sr != target_sr:
        sig_ = librosa.resample(sig, orig_sr=sr, target_sr=target_sr)

    S, phase = librosa.magphase(librosa.stft(sig_, hop_length=step_size, win_length=window_size))
    energy = librosa.feature.rms(S=S)

    mfcc_ = librosa.feature.mfcc(y=sig_, sr=target_sr, n_mfcc=12, dct_type=2, norm='ortho', hop_length=step_size, win_length=window_size)

    features = np.concatenate((mfcc_, energy))

    features_delta = librosa.feature.delta(features)
    features_delta2 = librosa.feature.delta(features, order=2)

    all_features = np.concatenate((features, features_delta, features_delta2))
    all_features = all_features.T
    return all_features

