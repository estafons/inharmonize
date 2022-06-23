import librosa
import numpy as np
import scipy
from scipy.optimize import least_squares
import cProfile
import time
from numba import jit


def model(x, u):
    return x[0] * u**3 + x[1]*u + x[2]

def fun(x, u, y):
    temp = model(x,u)
    return temp-y


def jac(x, u, y):
    J = np.empty((u.size, x.size))
    J[:, 0] = u**3
    J[:, 1] = u
    J[:, 2] = 1
    return J

def compute_least(u,y):
    x0=[0.00001,0.00001,0.000001]
    res = least_squares(fun, x0, jac=jac,bounds=(0,np.inf), args=(u, y),loss = 'soft_l1', verbose=0)
    return res.x


def computeInharmonicity(fundamental, diffs, maxOrder):
    u = np.arange(len(diffs)) + 2
    [a,b,_] =compute_least(u,diffs)
    beta = 2*a/(fundamental+b)
    return beta

class NoteClip():

    @property
    def fundamental(self):
        return self.__fundamental
    
    @fundamental.setter
    def fundamental(self, fundamental : float):
        self.fundamental = fundamental

    @staticmethod
    def loadAudio(filename : str):
        audio, sr = librosa.load(filename, sr = 44100)
        return audio, sr

    @staticmethod
    def computeDFT(audio, size_of_fft : int, sr : int):
        fft = np.fft.fft(audio,n = size_of_fft)
        frequencies = np.fft.fftfreq(size_of_fft,1/sr)
        return fft, frequencies

    @staticmethod
    def computePeak(fft, frequencies):
        """Method that computes max peak on given FFT. Will be used to
        detect a partial."""
        mY = np.abs(fft) # Find magnitude
        locY = np.argmax(mY) # Find maxpeak location
        return frequencies[locY] # Get the actual frequency value

    @staticmethod
    def filterAudio(fft, centrerFreq, windowSize, sr):
        """Method that does zone filtering arround centerFreq with WindowSize"""
        sz = fft.size
        x = np.zeros(sz,dtype=np.complex64)
        centerFreqY = int(round(centrerFreq*sz/sr))
        for i in range(centerFreqY - windowSize//2, centerFreqY + windowSize//2):
            x[i] = fft[i]
        return x

    
    @staticmethod
    def computeSTFT(audio):
        pass

    def __init__(self, fundamental, filename):
        self.__fundamental = fundamental
        self.audio, self.sr = self.loadAudio(filename)


class PartialComputer():
    
    @property
    def step(self):
        return self.__step
    
    @step.setter
    def step(self, step):
        self.__step = step

    @property
    def firstEval(self):
        return self.__firstEval
    
    @property
    def lastEval(self):
        return self.__lastEval
    
    @firstEval.setter
    def firstEval(self, firstEval):
        self.__firstEval = firstEval

    @lastEval.setter
    def lastEval(self, lastEval):
        self.__lastEval = lastEval

    @staticmethod
    def computePartialPeak(noteclip : NoteClip, fft, frequencies, centrerFreq, windowSize, sr):
        return noteclip.computePeak(noteclip.filterAudio(fft,
                                        centrerFreq, windowSize, sr), frequencies)
    
    @staticmethod
    def computeCenterfreq(fundamental : float, partialOrder : float, beta : float):
        return partialOrder*fundamental * np.sqrt(1 + beta*partialOrder**2)

    @staticmethod
    def computeTFreqs(fundamental, maxOrder):
        for i in range (2, maxOrder):
            yield i*fundamental

    def computePartial(self, noteclip : NoteClip, order, fft, frequencies, windowSize, sr, beta):
        centerFreq = self.computeCenterfreq(noteclip.fundamental, order, beta)
        return self.computePartialPeak(noteclip, fft, frequencies, centerFreq, windowSize, sr)

    @staticmethod
    def computeDiffs(mPartials, tPartials):
        return [m - t for m, t in zip(mPartials, tPartials)]

    # rewrite function for inner itteration. then write another for outer.
    def computeInnerIterPartials(self, noteclip, fft, frequencies, windowSize, sr, orderLimit, beta):
            for order in range(2, orderLimit):
                yield self.computePartial(noteclip, order, fft, frequencies, windowSize, sr, beta)
    
    


    def computeOuterIterPartials(self, noteclip, fft, frequencies, windowSize, sr):
        beta = 0
        for orderLimit in range(self.firstEval, self.lastEval, self.step):
            computeInnerIterGen = self.computeInnerIterPartials(noteclip, fft, frequencies, windowSize, sr, orderLimit, beta)
            diffs = self.computeDiffs(computeInnerIterGen, self.computeTFreqs(noteclip.fundamental, orderLimit))
            beta = computeInharmonicity(noteclip.fundamental, diffs, orderLimit)
        return beta

    def __init__(self, step, firstEval, lastEval):
        self.__step = step
        self.__firstEval = firstEval
        self.__lastEval = lastEval


if __name__ == "__main__":
    x = NoteClip(179, "175hz.wav")
    y = PartialComputer(2, 6, 50)
    fft, frequencies = x.computeDFT(x.audio, x.audio.size, x.sr)
    start_time = time.time()
    for i in range(1, 500):
        y.computeOuterIterPartials(x, fft, frequencies, 80, x.sr)

    print("--- %s seconds ---" % (time.time() - start_time))