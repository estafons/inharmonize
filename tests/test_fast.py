
from contexts import fast, convergenceChecker
import unittest
import numpy as np
import random
import librosa
import pytest

class TestFilter(unittest.TestCase):

    def test_filter_len(self):
        sr = 44100
        arrLen = random.randint(1, 10) * sr
        fft = np.ones(arrLen)
        frequencies = np.ones(arrLen)
        windowSize = 80
        cenFreq = random.random()*300
        f1, f2 = fast.filterAudio(fft, frequencies, cenFreq, windowSize, sr)
        self.assertEqual(len(f1), len(f2))
    
class TestFunctionality(unittest.TestCase):
    def test_result_accuracy(self):
        audio, sr = librosa.load("./src/175hz.wav", sr = 44100)
        fft, frequencies = fast.computeDFT(audio, audio.size, sr)
        beta = fast.computePartialsGenerator(179, 6, 50, 2, fft, frequencies, 80, sr, 32)
        temp = list(beta)[-1]

        assert pytest.approx(temp, 0.05) == 7.4381*10**(-5)


    def test_result_convergence(self):
        audio, sr = librosa.load("./src/175hz.wav", sr = 44100)
        fft, frequencies = fast.computeDFT(audio, audio.size, sr)
        beta = fast.computePartialsGenerator(179, 6, 50, 2, fft, frequencies, 80, sr, 32)
        check = convergenceChecker.checkConvergence(list(beta))
        assert check

if __name__ == "__main__":
    unittest.main()