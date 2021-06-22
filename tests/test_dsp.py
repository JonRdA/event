import types
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import dsp
import event

class TestEvent(unittest.TestCase):
    def setUp(self):
        # Correct file event.
        self.ev_1 = event.Event('../data/tests/2020.09.24_11.12.20.txt')
        self.ev_1.load_data()
        self.ev_1.select_reg(25000, 50001)

        self.size = 2**11
        self.win_gen = self.ev_1.windows(self.size)
        self.win = next(self.win_gen)
        self.c1 = self.win['c1']

        self.freq = dsp.fft_freq(self.size, self.ev_1.rate)
        self.mode_gen = dsp._mode_subset(self.c1, self.freq,
                                         self.ev_1.rate, self.size, 1.52)

        # Incorrect file event.
        self.ev_2 = event.Event('../test_data/no_data.txt')

        # Non existing file event.
        self.ev_3 = event.Event('myfile')

    def test_fft_freq(self):
        """Test returns a ndarray, has correct lenght, values
        are positive, max frequency is half the rate.
        """
        freq = dsp.fft_freq(self.size, self.ev_1.rate)
        freq2 = dsp.fft_freq(self.size, self.ev_1.rate)

        self.assertIsInstance(freq, np.ndarray)
        self.assertEqual(len(freq), self.size // 2 + 1)
        self.assertTrue((freq[1:] > 0).all())       # freq[0] = 0
        self.assertEqual(freq.max(), self.ev_1.rate / 2)

    def test_fft_amp(self):
        """Test returns an ndarray, has correct lenght, positive values
        and smaller than signal.
        """
        amp1 = dsp.fft_amp(self.c1, samples=self.size)
        amp2 = dsp.fft_amp(self.c1)

        self.assertTrue(np.equal(amp1, amp2).all())

        self.assertIsInstance(amp1, np.ndarray)
        self.assertEqual(len(amp1), self.size // 2 + 1)
        self.assertTrue((amp1 > 0).all())
        self.assertTrue(amp1.max() < self.c1.max())

    def test_apply_window(self):
        c1_win = dsp.apply_window(self.c1, samples=self.size, win_func='hann')

        self.assertEqual(len(c1_win), self.size)
        with self.assertRaises(ValueError):
            dsp.apply_window(self.c1, 300, win_func='badfunc')

    def test_rms(self):
        """Check rms works with and without samples, raises exception
        with 'nan' values and computes known values.
        """
        samps = 1000
        cycles = 10
        sine_signal = np.sin(np.arange(0, samps) / samps * 2 * cycles * np.pi)
        nan_signal = np.append(np.full(100, 8), np.full(1000, np.nan))

        rms_1 = dsp.rms(self.c1)
        rms_2 = dsp.rms(self.c1, samples=self.size)
        rms_3 = dsp.rms(sine_signal)

        self.assertTrue(np.equal(rms_1, rms_2).all())
        self.assertTrue(np.allclose(rms_3, np.sqrt(2) / 2))

        with self.assertRaises(ValueError):
            rms_3 = dsp.rms(nan_signal)

    def test_mode_subset(self):
        """Check returns a generator of correct size and types"""
        mode_gen = dsp._mode_subset(self.c1, self.freq,
                                    self.ev_1.rate, self.size, 1.52)
        mode_data = next(mode_gen)

        self.assertIsInstance(mode_gen, types.GeneratorType)
        self.assertIsInstance(mode_data, tuple)
        self.assertIsInstance(mode_data[0], str)
        self.assertIsInstance(mode_data[1], np.ndarray)
        self.assertIsInstance(mode_data[2], np.ndarray)
        self.assertEqual(len(mode_data[1]), len(mode_data[2]))
        # self.assertEqual(self.win.shape, (self.size, 4))

    def test_find_freqs(self):
        """Test output types , sizes and known values"""
        # Test decorator calculates samples with same result
        freqs_1 = dsp.find_freqs(self.c1, self.freq, self.ev_1.rate,
                                 1.52, samples=self.size, modes=[1, 2, 3])
        freqs_1_bis = dsp.find_freqs(self.c1, self.freq, self.ev_1.rate, 1.52,
                                     modes=[1, 2, 3])
        freqs_2 = dsp.find_freqs(self.c1, self.freq, self.ev_1.rate,
                                 1.52, modes=list(range(1, 6)))
        freqs_2_mln = [1.4744, 2.9948, 4.4865, 5.9614, 7.4376]

        freqs_3 = dsp.find_freqs(self.c1, self.freq, self.ev_1.rate, 1.52,
                                 samples=self.size, modes=[1, 2, 3],
                                 win_func='hann')
        freqs_3_mln = [1.4856, 2.9985, 4.4778]
        # print(freqs_3)

        self.assertIsInstance(freqs_1, pd.Series)
        self.assertTrue(np.allclose(freqs_1, freqs_1_bis))
        self.assertEqual(freqs_1.size, 3)
        self.assertEqual(freqs_2.size, 5)
        self.assertTrue(np.allclose(freqs_2_mln, freqs_2, rtol=1e-03))
        self.assertTrue(np.allclose(freqs_3_mln, freqs_3, rtol=1e-04))

    def test_find_amps(self):
        """Test output types , sizes and known values"""
        # Test decorator calculates samples with same result.
        amps_1 = dsp.find_amps(self.c1, self.freq, self.ev_1.rate, 1.52,
                               samples=self.size, min_amp=0, modes=[1, 2, 3])
        amps_1_bis = dsp.find_amps(self.c1, self.freq, self.ev_1.rate, 1.52,
                                   min_amp=0, modes=[1, 2, 3])

        # Test known values & decorator removes added 100 offset.
        amps_2 = dsp.find_amps(self.c1 + 100, self.freq, self.ev_1.rate, 1.52,
                               modes=list(range(1, 6)), rm_offset=True, min_amp=0)
        amps_2_mln = [7.988528, 0.737823, 4.681256, 0.8401647, 1.513304]

        self.assertIsInstance(amps_1, pd.Series)
        self.assertTrue(np.allclose(amps_1, amps_1_bis))
        self.assertEqual(amps_1.size, 3)
        self.assertEqual(amps_2.size, 5)
        self.assertTrue(np.allclose(amps_2_mln, amps_2, rtol=1e-06))

    def test_preprocess(self):
        """Test samples, remove offset, apply win and find nan."""
        amps_1 = dsp.find_amps(self.c1, self.freq, self.ev_1.rate, 1.52,
                               samples=self.size, min_amp=0, modes=[1, 2, 3])
        amps_2 = dsp.find_amps(self.c1, self.freq, self.ev_1.rate, 1.52,
                               min_amp=0, modes=[1, 2, 3])
        self.assertTrue(np.allclose(amps_1, amps_2))

        amps_3 = dsp.find_amps(self.c1 + 20, self.freq, self.ev_1.rate,
                               1.52, rm_offset=True, min_amp=0)
        amps_4 = dsp.find_amps(self.c1, self.freq, self.ev_1.rate, 1.52,
                               samples=self.size, min_amp=0)
        self.assertTrue(np.allclose(amps_3, amps_4))

        signal_one = np.full(1024, 3).astype(float)
        amplitudes = dsp.fft_amp(signal_one, win_func='hann')
        self.assertTrue(np.isclose(amplitudes[0], 3, rtol=1e-3))

        c_copy = self.c1.copy()     # To avoid warning of set value pd.
        c_copy.iloc[7] = np.nan
        with self.assertRaises(ValueError):
            amps_3 = dsp.find_amps(c_copy + 20, self.freq, self.ev_1.rate,
                                   1.52, rm_offset=True, find_nan=True)
            amps_3 = dsp.find_amps(self.c1, self.freq, self.ev_1.rate,
                                   1.52, win_func='badfunc')


if __name__ == '__main__':
    unittest.main()
