import itertools
import numpy as np
import time
import librosa
import numpy as np
import scipy
from scipy.optimize import least_squares
import cProfile
from numba import jit, vectorize, float64
from multiprocessing import Pool

@jit(nopython=True)
def model(x, u):
    return x[0] * u**3 + x[1]*u + x[2]

def fun(x, u, y):
    temp = model(x,u)
    return temp-y

@jit(nopython=True)
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

    


    def __init__(self, fundamental, filename):
        self.__fundamental = fundamental
        self.audio, self.sr = self.loadAudio(filename)



@jit(nopython=True)
def filterAudio(fft, frequencies, centrerFreq, windowSize, sr):
    """Method that does zone filtering arround centerFreq with WindowSize"""
    sz = fft.size
    centerFreqY = int(round(centrerFreq*sz/sr))
    filtered = fft[centerFreqY - windowSize//2:centerFreqY + windowSize//2].copy()
    fil_freqs = frequencies[centerFreqY - windowSize//2:centerFreqY + windowSize//2].copy()
    return filtered, fil_freqs

@jit(nopython=True)
def computePeak(fft, frequencies):
        """Method that computes max peak on given FFT. Will be used to
        detect a partial."""
        mY = np.abs(fft) # Find magnitude
        locY = np.argmax(mY) # Find maxpeak location
        return frequencies[locY] # Get the actual frequency value

@jit(nopython=True)
def computeCenterfreq(fundamental : float, partialOrder : float, beta : float):
    return partialOrder*fundamental * np.sqrt(1 + beta*partialOrder**2)

@jit(nopython=True)
def computeTFreqs(fundamental, maxOrder):
    return np.arange(2*fundamental, (maxOrder)*fundamental, fundamental)


def computePartialPeak(fft, frequencies, centrerFreq, windowSize, sr):
        return computePeak(*filterAudio(fft, frequencies,
                                        centrerFreq, windowSize, sr))


def computePartial(fundamental, order, fft, frequencies, windowSize, sr, beta):
    centerFreq = computeCenterfreq(fundamental, order, beta)
    return computePartialPeak(fft, frequencies, centerFreq, windowSize, sr)

@jit(nopython=True)
def computeDiffs(mPartials, tPartials):
    return np.subtract(mPartials, tPartials)

# @vectorize([float64(float64, float64)])
# def computeDiffs(x, y):
#     return x - y


def computeInnerIterPartials(fundamental, fft, frequencies, windowSize, sr, orderLimit, beta):
    x = np.empty(orderLimit-2)
    for order in range(2, orderLimit):
         x[order-2] = computePartial(fundamental, order, fft, frequencies, windowSize, sr, beta)
    return x

def computeOuterIterPartials(fundamental, firstEval, lastEval, step, fft, frequencies, windowSize, sr):
    beta = 0
    for orderLimit in range(firstEval, lastEval, step):
        computeInnerIterGen = computeInnerIterPartials(fundamental, fft, frequencies, windowSize, sr, orderLimit, beta)
        diffs = computeDiffs(computeInnerIterGen, computeTFreqs(fundamental, orderLimit))
        beta = computeInharmonicity(fundamental, diffs, orderLimit)
    # print(beta)
    return beta


def func(fundamental, fft, frequencies, sr):
    # print("itsfunc")
    computeOuterIterPartials(fundamental,6, 50, 2, fft, frequencies, 80, sr)
    

if __name__ == "__main__":
    x = NoteClip(179, "175hz.wav")
    # y = PartialComputer(2, 6, 50)
    fft, frequencies = x.computeDFT(x.audio, x.audio.size, x.sr)
    start_time = time.time()

    testObj = type('testType', (object,), 
                 {'fundamental' : 179, 'fft' : fft, 'frequencies' : frequencies, 'sr' : x.sr})()
    
    itero = itertools.repeat((x.fundamental, fft, frequencies, x.sr), 500)
    # print(list(map(func, itertools.repeat((x.fundamental, fft, frequencies, x.sr), 50))))
    with Pool(5) as p:
        p.starmap(func, itero)
    # func(x.fundamental, fft, frequencies, x.sr)
    print("--- %s seconds ---" % (time.time() - start_time))