# from scikits.talkbox.features import mfcc
# import scipy
# from scipy.io import wavfile

import librosa
import numpy as np


def convert_to_mfcc(voice_path):
    y, _ = librosa.load(voice_path)
    mfccs = librosa.feature.mfcc(y=y, n_mfcc=20)


    result = np.zeros((len(mfccs[0]), 20))
    for i, mfcc in enumerate(mfccs):
        for j, value in enumerate(mfcc):
            result[j, i] = value

    return result
