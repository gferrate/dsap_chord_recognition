import os
from scipy.io import wavfile
import numpy as npy
from math import log10
import matplotlib.pyplot as plot
import aifc
from scipy import signal
from skimage.feature import peak_local_max
from numpy import fft
import matplotlib.collections as collections
import librosa
import sounddevice as sd
import sounddevice as sd
import numpy as np

class PCPExtractor:

    def __init__(self, window_size, fs, fref=261.63, min_dist_maxima_spectrum=5, min_freq_threshold=100):
        self.window_size=window_size
        self.fs=fs
        self.fref=fref
        self.min_dist_maxima_spectrum = min_dist_maxima_spectrum
        self.min_freq_threshold = min_freq_threshold
        self.NOTES = [
            'do',     'do#',    're',     're#',    'mi',     'fa',
            'fa#',    'sol',    'sol#', 'la',     'la#',    'si'
        ]

    def compute_fft(self, signal):
        spectrum = abs(fft.fft(signal))
        return spectrum

    def filter_fft(self, spectrum):
        indexes = peak_local_max(spectrum, min_distance=self.min_dist_maxima_spectrum)
        filtered_spectrum = npy.zeros(len(spectrum))
        for i in indexes:
            real_f_hz = self.fs * i / self.window_size
            if real_f_hz > self.min_freq_threshold:
                filtered_spectrum[i] = spectrum[i]
        return filtered_spectrum

    def calculate_PCP(self, filtered_spectrum, normalize_values=True):
        mapping = npy.zeros(int(self.window_size/2)).astype(int)
        mapping[0] = -1
        PCP = npy.zeros(12)

        for x in range(1, int(self.window_size/2)):
            real_f_hz = self.fs * x / self.window_size
            mapping[x] = int(
                    npy.mod(round(12 * npy.log2(real_f_hz / self.fref)), 12)
            )
            PCP[mapping[x]] += npy.square(filtered_spectrum[x])

        if normalize_values:
            PCP = PCP / max(PCP)

        PCP = dict(zip(self.NOTES, PCP))
        return PCP

    def get_PCP(self, windowed_data):
        pcp = self.calculate_PCP(self.filter_fft(self.compute_fft(windowed_data)))
        return pcp


class ChordMatcher:

    def __init__(self, threshold=1.8, method = 'threshold'):
        self.threshold = threshold
        self.method = method
        notes = [
            'do',     'do#',    're',        're#',     'mi',     'fa',
            'fa#',    'sol',    'sol#',    'la',        'la#',    'si'
        ]
        self.CHORDS = {
            'Do Maj':     dict(zip(notes, [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])),
            'Do# Maj':    dict(zip(notes, [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0])),
            'Re Maj':     dict(zip(notes, [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0])),
            'Re# Maj':    dict(zip(notes, [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0])),
            'Mi Maj':     dict(zip(notes, [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1])),
            'Fa Maj':     dict(zip(notes, [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0])),
            'Fa# Maj':    dict(zip(notes, [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0])),
            'Sol Maj':    dict(zip(notes, [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1])),
            'Sol# Maj': dict(zip(notes, [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0])),
            'La Maj':     dict(zip(notes, [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0])),
            'La# Maj':    dict(zip(notes, [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0])),
            'Si Maj':     dict(zip(notes, [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1])),

            'Do Min':     dict(zip(notes, [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])),
            'Do# Min':    dict(zip(notes, [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])),
            'Re Min':     dict(zip(notes, [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0])),
            'Re# Min':    dict(zip(notes, [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0])),
            'Mi Min':     dict(zip(notes, [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1])),
            'Fa Min':     dict(zip(notes, [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0])),
            'Fa# Min':    dict(zip(notes, [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0])),
            'Sol Min':    dict(zip(notes, [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0])),
            'Sol# Min': dict(zip(notes, [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1])),
            'La Min':     dict(zip(notes, [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0])),
            'La# Min':    dict(zip(notes, [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])),
            'Si Min':     dict(zip(notes, [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1])),
        }

    def compute_correlation(self, PCP):
        correlation = {}
        for chord, pcp_dict in self.CHORDS.items():
            correlation[chord] = 0
            for note, pcp_value in PCP.items():
                correlation[chord] += pcp_value * pcp_dict[note]

        return correlation

    def find_chord(self, correlation):
        sorted_correlation = sorted(
            correlation.items(), key=lambda kv: kv[1], reverse=True
        )
        chord = None
        if self.method == 'threshold':
            THRESHOLD = self.threshold
            max_value = sorted_correlation[0][1]
            if max_value > THRESHOLD:
                chord = sorted_correlation[0][0]
        elif self.method == 'difference':
            THRESHOLD = self.threshold
            ratio = sorted_correlation[0][1]/sorted_correlation[1][1]
            if ratio > THRESHOLD:
                chord = sorted_correlation[0][0]

        return chord, max_value

    def get_chord(self, PCP, silence):
        if silence:
            return None
        else:
            chord, max_value = self.find_chord(self.compute_correlation(PCP))
            return chord


def get_power(windowed_data, fs):
    # f, Pxx_spec = signal.periodogram(windowed_data, fs, 'flattop', scaling='spectrum')
    # power = npy.sqrt(Pxx_spec.max())
    power = 0
    for i in range(0, len(windowed_data)):
        power += windowed_data[i]**2
    power = power / len(windowed_data)
    return power


duration = 200  # seconds
fs = 44100
window_size = 1024*10

pcp_extractor = PCPExtractor(
    window_size=window_size,
    fs=fs,
    min_freq_threshold=200
)

chord_matcher=ChordMatcher(
    #threshold=2.2,
    threshold=2,
)
last_chord = None
chord_conversion = {
    'Do Maj': 'C',
    'Do# Maj': 'C#',
    'Re Maj': 'D',
    'Re# Maj': 'D#',
    'Mi Maj': 'E',
    'Fa Maj': 'F',
    'Fa# Maj': 'F#',
    'Sol Maj': 'G',
    'Sol# Maj': 'G#',
    'La Maj': 'A',
    'La# Maj': 'A#',
    'Si Maj': 'B',
    'Do Min': 'C:min',
    'Do# Min': 'C#:min',
    'Re Min': 'D:min',
    'Re# Min': 'D#:min',
    'Mi Min': 'E:min',
    'Fa Min': 'F:min',
    'Fa# Min': 'F#:min',
    'Sol Min': 'G:min',
    'Sol# Min': 'G#:min',
    'La Min': 'A:min',
    'La# Min': 'A#:min',
    'Si Min': 'B:min'
}
convert_to_letters = False

def mic_callback(indata, outdata, frames, time, status):
    global pcp_extractorm, chord_matcher, last_chord
    left_channel = indata[:, 0] # Left channel
    windowed_data = signal.windows.cosine(window_size) * left_channel
    power = get_power(windowed_data, fs)
    if power > 1e-07:
        PCP = pcp_extractor.get_PCP(windowed_data)
        chord = chord_matcher.get_chord(PCP, silence=False)
        if chord and chord != last_chord:
            if convert_to_letters:
                chord = chord_conversion[chord]
            print(f'Chord: {chord}')
            last_chord = chord


with sd.Stream(callback=mic_callback, blocksize=window_size, samplerate=fs):
    sd.sleep(duration * 1000)
