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
    """ 
    
    Function that computes non linear least squares.

    Parameters
    ------------
    u : 1-dimension array
      Orders of partials.
    y : 1-dimension array
      differences measured.

    Returns
    -----------
    tuple of 3 float. Then used to compute beta coefficient.


    """
    x0=[0.00001,0.00001,0.000001]
    res = least_squares(fun, x0, jac=jac,bounds=(0,np.inf), args=(u, y),loss = 'soft_l1', verbose=0) 
    return res.x


def computeInharmonicity(fundamental, diffs, maxOrder):
    """
    
    Function that computes inharmonicity coefficient.
    
    Parameters
    -----------
    fundamental : float
                fundamental frequency of note.
    diffs : 1-dimension array

    Returns
    ----------
    beta : float
    beta coefficient computed

    """
    u = np.arange(len(diffs)) + 2
    [a,b,_] =compute_least(u,diffs)
    beta = 2*a/(fundamental+b)
    return beta


@jit(nopython=True)
def filterAudio(fft, frequencies, centrerFreq, windowSize, sr):
    """Method that does zone filtering arround centerFreq with WindowSize
    
    Parameters
    ------------
    fft : 1-dimension array
        fft of input signal
    frequencies : 1-dimension array
                frequencies of fft
    centerFreq : float
               frequency arround which zone filtering is performed.
    windowSize : float
               Size of zone filter
    sr : int
        fft sampling rate
    
    Returns
    -----------
    filtered : 1-dimension array
             part of the fft arround centerFreq
    fil_freqs : 1-dimension array
              corresponding frequencies to filtered fft

    """
    sz = fft.size
    centerFreqY = int(round(centrerFreq*sz/sr))
    filtered = fft[centerFreqY - windowSize//2:centerFreqY + windowSize//2].copy()
    fil_freqs = frequencies[centerFreqY - windowSize//2:centerFreqY + windowSize//2].copy()
    return filtered, fil_freqs

@jit(nopython=True)
def computePeak(fft, frequencies):
    """Method that computes max peak on given FFT. Will be used to
    detect a partial.
    
    Parameters
    -----------

    fft : 1-dimension array
    frequencies : 1-dimension array

    Returns
    -----------
    peak : float
            max peak of fft in Hz

    """
    mY = np.abs(fft) # Find magnitude
    locY = np.argmax(mY) # Find maxpeak location
    return frequencies[locY] # Get the actual frequency value

@jit(nopython=True)
def computeCenterfreq(fundamental : float, partialOrder : int, beta : float):
    """
    Centering func for filtering fft based on beta computed so far.

    Parameters
    ----------
    fundamental : float
    partialOrder : int
                 order of partial searching
    beta : float
          beta coefficient computed so far.
    
    Returns
    ---------
    centerFreq : float
               frequency center to search for partial based on beta.
                  
    """
    return partialOrder*fundamental * np.sqrt(1 + beta*partialOrder**2)

@jit(nopython=True)
def computeTFreqs(fundamental, maxOrder):
    """
    Computing theoretical partials (k*f_0)

    Parameters
    ----------
    fundamental : float
    maxOrder : int
                 max order of partial to compute
    
    Returns
    ---------
    tPartials : 1-dimension array
               theoretical partials expected based on fundamental
                  
    """
    return np.arange(2*fundamental, (maxOrder)*fundamental, fundamental)


def computePartialPeak(fft, frequencies, centrerFreq, windowSize, sr):
    """
    Compute peak of given partial.

    fft : 1-dimension array
        fft of input signal
    frequencies : 1-dimension array
                frequencies of fft
    centerFreq : float
               frequency arround which zone filtering is performed.
    windowSize : float
                Size of zone filter
    sr : int
       fft sampling rate
    
    Returns
    ---------
    peak : float
            max peak of fft/partial in Hz
                    
    """
    return computePeak(*filterAudio(fft, frequencies,
                                        centrerFreq, windowSize, sr))


def computePartial(fundamental, order, fft, frequencies, windowSize, sr, beta):
    """
    Find partial with order.

    fundamental : float
    order : int
           order of partial
    fft : 1-dimension array
        fft of input signal
    frequencies : 1-dimension array
                frequencies of fft
    centerFreq : float
               frequency arround which zone filtering is performed.
    windowSize : float
                Size of zone filter
    sr : int
       fft sampling rate
    beta : float
         beta coefficient computed so far.
    
    Returns
    ---------
    peak : float
            max peak of fft/partial in Hz
                    
    """
    centerFreq = computeCenterfreq(fundamental, order, beta)
    return computePartialPeak(fft, frequencies, centerFreq, windowSize, sr)

@jit(nopython=True)
def computeDiffs(mPartials, tPartials):
    """
    Compute peak of given partial.

    mPartials : 1-dimension array
        measured partials.
    tPartials : 1-dimension array
              ideal/theoretical partials' frequencies
    
    Returns
    ---------
    diffs : 1-dimension array
          differences between beasured and ideal partials.
                    
    """
    return np.subtract(mPartials, tPartials)

def computeInnerIterPartials(fundamental, fft, frequencies, windowSize, sr, orderLimit, beta):
    """
    Compute partials up to a limit orderLimit with a given beta.

    fundamental : float
    fft : 1-dimension array
    frequencies : 1-dimension array
                 frequencies associated to fft
    windowSize : float
               window size selected to filter arround each partial search window.
    sr : int
        sampling rate
    orderLimit : int
                up to number of partials to compute
    beta : float
         beta computed so far
    
    Returns
    ---------
    mPartials : 1-dimension array
               partials computed so far
                    
    """
    x = np.empty(orderLimit-2)
    for order in range(2, orderLimit):
         x[order-2] = computePartial(fundamental, order, fft, frequencies, windowSize, sr, beta)
    return x

def computeOuterIterPartials(fundamental, firstEval, lastEval, step, fft, frequencies, windowSize, sr):
    """
    Compute partials and beta coefficient.

    fundamental : float
    firstEval : int
              first order of partial that beta will be evaluated for the first time.
    lastEval : int
             number of partials to use for beta computation.
    step : int
         step of order of partials to compute beta.
    fft : 1-dimension array
    frequencies : 1-dimension array
                 frequencies associated to fft
    windowSize : float
               window size selected to filter arround each partial search window.
    sr : int
        sampling rate
    
    Returns
    ---------
    beta : float
          beta computed for this note instance
                    
    """
    beta = 0
    for orderLimit in range(firstEval, lastEval, step):
        computeInnerIterGen = computeInnerIterPartials(fundamental, fft, frequencies, windowSize, sr, orderLimit, beta)
        diffs = computeDiffs(computeInnerIterGen, computeTFreqs(fundamental, orderLimit))
        beta = computeInharmonicity(fundamental, diffs, orderLimit)
        #print(beta)
    return beta

