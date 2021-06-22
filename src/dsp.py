"""Digital Signal Processing module.

Handles DSP operations with real 1d signal as input.

functions decorated with @preprocess accept keyword arguments:
    samples: calculate signal size if not given.
    rm_offset: remove offset of signal substracting the mean.
    find_na: raise ValueError if signal has nan.
"""

import logging
import numpy as np
import pandas as pd
from scipy import signal as sgn
from functools import wraps
import matplotlib.pyplot as plt

import event

logger = logging.getLogger(__name__)

# Decorators

def preprocess(func):
    """Decorator to optionally preprocess parameters on DSP functions.

    Preprocessing implemented through keyword arguments:
        samples: calculate signal size if not given.
        rm_offset: remove offset of signal substracting the mean.
        find_na: raise ValueError if signal has nan.
        win_func: windowing function to multiply the signal.

    The first argument of the function must be 'signal'.

    If 'samples', size of the signal, is passed, does nothing.
    Otherwise, computes it and passes it as kwarg to the function.

    If keyword argument rm_offest is true, it subtracts the mean of
    the signal to itself and passes is as argument.

    Args:
        func (function): DSP function that operates with signal size.

    Returns:
        fuction: function called with 'samples' argument.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        """Inner func where functionalities are added."""

        # Check if any keyword arg was passed to preprocess.
        # print(f"Entering function 'wrapper' witn kwargs '{kwargs}'")
        kk = kwargs.keys()                      # Keyword argument keys.
        conds = [not 'samples' in kk]           # Conditions.
        possible_kwargs = ['rm_offset', 'find_na', 'win_func']
        conds += [kw_arg in kk for kw_arg in possible_kwargs]

        # 'samples'  changes kwargs, rest change 'signal'
        if any(conds):
            num_args = len(args)
            signal = args[0]

            # If 'samples' not passed, compute size.
            if 'samples' not in kk:
                samples = len(signal)
                kwargs['samples'] = samples

            if 'find_nan' in kk:
                if np.isnan(signal).any():
                    raise ValueError("'signal' has nan values")

            # Options that modify signal, reconstruct args tuple.
            if any([x in kk for x in ['rm_offset', 'win_func']]):
                if kwargs.get('rm_offset'):
                    signal -= signal.mean()

                # Check that True values were
                if kwargs.get('win_func'):
                    samples = kwargs['samples']
                    win_func = kwargs['win_func']
                    signal = apply_window(signal, samples, win_func)

                # Reconstruct args, keep remaining ones if exits.
                if num_args > 1:
                    *rest_args, = args[1:]
                    args = (signal, *rest_args)
                else:
                    args = (signal,)
        # Return function called with modified (or not) args, kwargs.
        # print(f"Entering function '{func.__name__}' witn kwargs '{kwargs}'")
        # print(f"Entering function '{func.__name__}' witn args len '{len(args)}'")
        return func(*args, **kwargs)

    # Return inner function.
    return wrapper


# Digital Signal Processing

def fft_freq(samples, rate):
    """Compute the frequency bins of real acceleration input FFT

    Args:
        samples (int): signal size
        rate (int): signal measuring rate

    Returns:
        np.ndarray: frequencies
    """
    freq = np.fft.rfftfreq(samples, 1 / rate)
    return freq

def fft_zoom(freq, f0, f1):
    """Calculate index to select a frequency subset [f0, f1]

    Args:
        freq (np.ndarray): frequency array
        f0 (TYPE): initial freqency value, inclusive
        f1 (TYPE): final frequency value, inclusive

    Returns:
        tuple: Description
    """
    freq_res = freq[1] - freq[0]
    i0 = int(f0 / freq_res)
    i1 = int(f1 / freq_res) + 2
    return i0, i1

@preprocess
def fft_amp(signal, samples=0, **kwargs):
    """Computes the FFT of real acceleration input.

    Returns amplitudes of the frequency domain components of the signal.

    Args:
        signal (np.ndarray): 1 dimensional real data input.
        samples (int, optional): size of signal, default is 0.

    Returns:
        np.ndarray: amplitude values.
    """
    # *2 because 2 complex waves compose the real one
    # and / n to normalize. DFT adds all signal
    amp = np.abs(np.fft.rfft(signal)) * 2 / samples
    return amp

def apply_window(signal, samples, win_func, **kwargs):
    """Apply specified window to the signal by multiplying it.

    No windowing if invalid 'win_func' and warning will be logged.

    Args:
        signal (np.ndarray): 1d real signal of floats.
        samples (int, optional): size of signal, decorator can provide.
        win_func (str, optional): window function to be applied.
          supported: ['hann', 'flattop']
    Returns:
        np.ndarray: windowed signal
    """
    if win_func == 'hann':
        signal *= sgn.hann(samples)
    elif win_func == 'flattop_to be properly implemented':
        signal *= sgn.flattop(samples) * 4.636      # Merlin does that, ask. some work on the coeficients is needed
        # signal *= sgn.flattop(samples)
    else:
        raise ValueError(f"Unsupported func '{win_func}' no window was applied")

    return signal

@preprocess
def rms(signal, samples=0, **kwargs):
    """Computes the Root Mean Square of the signal.

    Available option to balance the offset by removing the mean.

    Args:
        signal (np.ndarray): 1d real signal.
        samples (int, optional): size of signal.
        offset (bool, optional): remove signal offset. Default is False.

    Returns:
        float

    Raises:
        ValueError: if signal contains 'nan' values.
    """
    squared = np.square(np.asarray(signal))
    rms = np.sqrt(np.sum(squared) / samples)

    if np.isnan(rms):
        raise ValueError("signal has 'nan' values.")

    return rms

# Amplitude and frequency finding

def _mode_subset(signal, freq, rate, main_freq, samples, modes=[1], width=0.2):
    """Yields frequency domain data subsets around modes.

    To be used on 'find_freqs' and 'find_amps' functions.

    Args:
        signal (np.ndarray): 1 dimensional real data input.
        freq (np.ndarray): signal frequency array.
        rate (int): signal measuring rate.
        main_freq (float): main frequency to find.
        samples (int): signal size.
        modes (list): integer modes (multiples) around to subset.
        width (float, optional): % of the freq_mode wher to search.

    Yields:
        tuple: (mode name, freq_subset, amp_subset)
    """
    # Compute the FFT.
    amp = fft_amp(signal, samples=samples)

    # Calculate resolution in frequency domain.
    res = (freq[1] - freq[0])

    for mode in modes:
        m_name = f'm{mode}'

        # Search limits indexes
        lower = int(round(main_freq * (mode - width) / res))
        upper = int(round(main_freq * (mode + width) / res))

        yield m_name, freq[lower: upper], amp[lower: upper]

def _find_amp(amp, min_amp):
    """Finds the maximum amplitude value.

    Signal's subsets must be passed. If amplitude value smaller than
    treshold will return 'nan'.

    Args:
        amp (np.ndarray): signal amplitude values subset.
        min_amp (float): amplitude treshold to retrieve value.

    Returns:
        float: max amp value.
    """
    max_val = np.max(amp)
    return max_val if max_val > min_amp else np.nan

def _find_freq(freq, amp):
    """Finds the frequency value of the maximum amplitude.

    Subsets of the original signal must be passed.
    The bin of maximum amplitude is averaged with is neighbors using
    a weighed mean of 3 bins. If no relative maximum is found on the
    passed subset (max is on edges) logs warning and returns nan.

    Args:
        freq (np.ndarray): signal frequency array subset.
        amp (np.ndarray): signal amplitude values subset.

    Returns:
        float: frequency value.
    """
    # Find where the max amplitude is. If it's on an edge, warn & nan.
    max_index = np.argmax(amp)
    if max_index < 1 or max_index > len(amp):
        freq_win = [round(freq[0], 2), round(freq[1], 2)]
        logger.warning(f'no peak in: {freq_win}')
        return np.nan

    # Average using 3 bins.
    avg_freq = freq[max_index - 1: max_index + 2]
    avg_amp = amp[max_index - 1: max_index + 2]

    avg = np.average(avg_freq, weights=avg_amp)
    return avg

@preprocess
def find_freqs(signal, freq, rate, main_freq,
               samples=0, modes=[1, 2], width=.2, **kwargs):
    """Find signals' main frequency and modes' peak frequency values.

    An approximate 'main_freq' value must be given to guide search.

    Args:
        signal (np.ndarray): 1 dimensional real data input.
        freq (np.ndarray): signal frequency array.
        rate (int): signal measuring rate.
        samples (int): signal size.
        main_freq (float): main frequency to find.
        modes (list): integer modes (multiples) to find freq.
        width (float, optional): % of the freq_mode wher to search.

    Returns:
        pd.Series: mode peak frequency values.
    """
    result = pd.Series(dtype='float64')

    for m_name, f, a in _mode_subset(signal, freq, rate, main_freq,
                                     samples, modes, width):
        result[m_name] = _find_freq(f, a)

    return result

@preprocess
def find_amps(signal, freq, rate, main_freq,
              samples=0, modes=[1, 2], width=.2, min_amp=5, **kwargs):
    """Find signals' main frequency and modes' peak amplitude values.

    An approximate 'main_freq' value must be given to guide search.
    If modes > 1 finds peaks of higher modes, multiples. A weighted
    mean of 3 bins is used.

    Args:
        signal (np.ndarray): 1 dimensional real data input.
        freq (np.ndarray): signal frequency array.
        rate (int): signal measuring rate.
        samples (int): signal size.
        main_freq (float): main frequency to find.
        modes (list): integer modes (multiples) to find amps.
        width (float, optional): % of the freq_mode wher to search.

    Returns:
        pd.Series: mode peak frequency values per channels.
    """
    result = pd.Series(dtype='float64')

    for m_name, f, a in _mode_subset(signal, freq, rate, main_freq,
                                     samples, modes, width):
        result[m_name] = _find_amp(a, min_amp)

    return result


def main():

    size = int(1024)


if __name__ == '__main__':

    # If module directly run, load log configuration for all modules.
    import logging.config
    logging.config.fileConfig('../log/logging.conf')
    logger = logging.getLogger('dsp')

    main()
