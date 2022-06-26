from multiprocessing import Pool
import numpy as np
from fast import computeOuterIterPartials
import itertools
import librosa
import time

def computeDFT(audio, size_of_fft : int, sr : int):
        fft = np.fft.fft(audio,n = size_of_fft)
        frequencies = np.fft.fftfreq(size_of_fft,1/sr)
        return fft, frequencies


def splitAtTimes(audio, times, sr):
    for time in times:
        yield np.array(audio[round(time["start"] * sr): round(time["end"] * sr)])


def DFTandInha(fundamental, firstEval, lastEval, step, audio, windowSize, sr):
    fft, frequencies = computeDFT(audio, audio.size, sr)
    computeOuterIterPartials(fundamental, firstEval, lastEval, step, fft, frequencies, windowSize, sr)

def createIter(fundamentals, firstEval, lastEval, step, audio,times, windowSize, sr):
    audios = splitAtTimes(audio, times, sr)
    return zip(fundamentals, itertools.repeat(firstEval), itertools.repeat(lastEval), itertools.repeat(step),
         audios, itertools.repeat(windowSize), itertools.repeat(sr))

def computeInhs(itero):
    with Pool(5) as p:
        p.starmap(DFTandInha, itero, chunksize=10)
filename = "/home/estfa/Projects/inharmonize/data/train/firebrand1/string2/2.wav"
audio, sr = librosa.load(filename, sr = 44100)


def wrapp():
    times = []
    filename = "/home/estfa/Projects/inharmonize/data/train/firebrand1/string2/2.wav"
    audio, sr = librosa.load(filename, sr = 44100)
    times.extend(itertools.repeat({"start": 0, "end" : 1}, 1000))
    times.append({"start": 0, "end" : 1})
    computeInhs(createIter([123 for x in range(1000)], 6, 50, 2, audio, times, 60, sr))


start_time = time.time()

    
# y = PartialComputer(2, 6, 50)
wrapp()
print("--- %s seconds ---" % (time.time() - start_time))