import unittest
import types
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import event

class TestEvent(unittest.TestCase):

    def setUp(self):
        # Correct file event.
        self.ev_1 = event.Event('../data/tests/2020.09.24_11.12.20.txt')
        self.ev_1.load_data()

        self.size = 1024
        self.gen = self.ev_1.windows(self.size)
        self.win = next(self.gen)
        self.c1 = self.win['c1']

        # Incorrect file event.
        self.ev_2 = event.Event('../data/tests/no_data.txt')

        # Non existing file event.
        self.ev_3 = event.Event('bad_file_path')

    def test_init(self):
        # Event objects created.
        self.assertIsInstance(self.ev_1, event.Event)
        self.assertIsInstance(self.ev_2, event.Event)
        self.assertIsInstance(self.ev_3, event.Event)

        with self.assertRaises(AttributeError):
            self.ev_1.data
            self.ev_2.data
            self.ev_3.data
            self.ev_1.rate

    # Data loading

    def test_load_data(self):
        # Data loaded in df object.
        self.assertIsInstance(self.ev_1.data, pd.DataFrame)

        # Correct loaded data format.
        cols = ['t', 'c1', 'c2', 'c3']
        self.assertEqual(list(self.ev_1.data.columns), cols)

        # Can not load on Event 2 and 3.
        with self.assertRaises(ValueError):
            self.ev_2.load_data()

        with self.assertRaises(FileNotFoundError):
            self.ev_3.load_data()
        # self.assertRaises(FileNotFoundError, self.ev_3.load_data())

    def load_info(self):
        self.ev_1.load_info('../test_data/event_amp_info.csv')

        self.assertEqual(self.ev_1.info.index[0][0], 'c')

        with self.assertRaises(FileNotFoundError):
            self.ev_3.load_info('test_input/badfile.txt')

    # Setters

    def test_set_info(self):
        """Test wrong data type, and wrong format raises errors."""
        with self.assertRaises(TypeError):
            self.ev_1.set_info(pd.DataFrame([1, 2, 2]))
            self.ev_1.set_info('fdsa')

        # Correct data type but wrong format.
        with self.assertRaises(ValueError):
            self.ev_1.set_info(pd.DataFrame([1, 2], index=['c1', 'fc3']))

    def test_set_rate(self):
        self.assertEqual(self.ev_1.rate, 200)

    def test_set_datetime(self):
        """Test creates correct attributes and raises errors."""
        self.ev_1.set_datetime()
        self.assertIsInstance(self.ev_1.dt_0, datetime.datetime)
        self.assertIsInstance(self.ev_1.dt_1, datetime.datetime)
        self.assertIsInstance(self.ev_1.d_t, datetime.timedelta)
        self.assertEqual(self.ev_1.dt_1 - self.ev_1.dt_0, self.ev_1.d_t)

        with self.assertRaises(ValueError):
            self.ev_2.set_datetime()
            self.ev_3.set_datetime()
        with self.assertRaises(AttributeError):
            self.ev_2.set_datetime(datetime.datetime(2021, 12, 3, 4, 5, 12))

    def test_set_dt_index(self):
        """Test creates correct index and raises error."""
        self.ev_1.set_dt_index()
        self.assertEqual(self.ev_1.data.index.dtype, 'datetime64[ns]')
        self.assertEqual(self.ev_1.data.index.name, 'dt')

        with self.assertRaises(AttributeError):
            self.ev_2.set_dt_index(datetime.datetime(2021, 12, 3, 4, 5, 12))

    # Data selection

    def test_windows(self):
        """Test returns a generator with dataframe of correct size."""
        self.assertIsInstance(self.gen, types.GeneratorType)
        self.assertIsInstance(self.win, pd.DataFrame)
        self.assertEqual(self.win.shape, (self.size, 4))

        # Test the mean removing functionality
        # Mean is zero if applied.
        gen2 = self.ev_1.windows(self.size, rm_offset=True)
        win2 = next(gen2)
        means = win2.mean().iloc[1:]
        zeros = np.array([0, 0, 0])
        self.assertTrue(np.allclose(means, zeros))

        # Add ofset and remove with rm_offset, compare known values.
        self.ev_1.data.iloc[:, 1] += 100
        gen3 = self.ev_1.windows(self.size, rm_offset=True)
        win3 = next(gen2)
        means2 = win3.mean().iloc[1:]
        zeros2 = [100.07, 0.09, 0.009]
        self.assertTrue(np.allclose(means2, zeros2, rtol=1e-01))

    def test_select_reg(self):
        """Test invalid input, appending zeros and known values."""
        with self.assertRaises(IndexError):
            self.ev_1.select_reg(80, 6)
            self.ev_1.select_reg(-100, 6)
            self.ev_1.select_reg(100000, 110000, False)
            self.ev_1.select_reg(100000, 110000, True)

        self.ev_1.select_reg(101, 201)
        self.assertEqual(self.ev_1.data.shape[0], 100)

        self.ev_1.select_reg(151, 161)
        self.assertEqual(self.ev_1.data.shape[0], 10)

        self.ev_1.select_reg(156, 166, True)
        self.assertEqual(self.ev_1.data.shape[0], 10)

    def test_select_ch(self):
        """Test invalid input, and known values"""

        orig_shape = self.ev_1.data.shape

        with self.assertRaises(TypeError):
            self.ev_1.select_ch(1)

        with self.assertRaises(KeyError):
            self.ev_1.select_ch(['jkl'])
            self.ev_1.select_ch('c1')

        self.ev_1.select_ch(['c1'])
        self.assertEqual(self.ev_1.data.shape, (orig_shape[0], 2))

    def test_select_vibration(self):
        """Test first reg increases and last reg no, also with delay"""
        last_reg = self.ev_1.data.shape[0] - 1
        self.ev_1.select_vibration(0)

        self.assertEqual(self.ev_1.data.index[0], 20630)
        self.assertEqual(self.ev_1.data.index[-1], last_reg)

        self.ev_1.load_data()
        self.ev_1.select_vibration(10)
        self.assertEqual(self.ev_1.data.index[0], 20630 + 2000)
        self.assertEqual(self.ev_1.data.index[-1], last_reg)

        self.ev_1.load_data()
        self.ev_1.select_vibration(5, ch='c2')
        self.assertEqual(self.ev_1.data.index[0], 20122 + 1000)
        self.assertEqual(self.ev_1.data.index[-1], last_reg)

        data = self.ev_1.data

    # Event concatenation

    def test_join(self):
        """Test overlapping and gap event joining. Assert total db size
        duration, start and end times.
        """
        ev1 = event.Event('../test_data/join_files/2020.01.01_00.00.10.txt')
        ev2 = event.Event('../test_data/join_files/2020.01.01_00.00.20.txt')
        ev3 = event.Event('../test_data/join_files/2019.09.24_11.12.20.txt')
        ev4 = event.Event('../test_data/join_files/2019.09.24_11.12.21.txt')

        events = [ev1, ev2, ev3, ev4]
        [i.load_data() for i in events]

        # Overlapping events
        jev1 = ev1.join(ev2)
        self.assertEqual(jev1.data.shape, (6000, 4))
        self.assertTrue((jev1.data.index == range(1, 6001)).all())
        self.assertTrue(np.allclose(jev1.data['t'], np.arange(0, 30, .005)))

        # No overlapping events
        jev2 = ev3.join(ev4)
        self.assertEqual(jev2.data.shape, (20, 4))
        self.assertTrue(jev2.data.iloc[-1, 0], 1.045)
        self.assertTrue(jev2.data.iloc[-1, 0], jev2.d_t)
        self.assertTrue(jev2.dt_0, ev3.dt_0)

    def test_concat(self):
        """Test concatenation with gap and with overlap. Check initial
        ending and duration datetimes. Check index continuity.
        """
        # Non overlapping events.
        path_files = Path('../test_data/join_files/')
        file_names = ['2019.09.24_11.12.20.txt',
                      '2019.09.24_11.12.21.txt',
                      '2019.09.24_11.12.22.txt',
                      '2019.09.24_11.12.23.txt']
        events = []
        for file in file_names:
            ev = event.Event(Path(path_files, file))
            ev.load_data
            events.append(ev)
        joined_1 = event.concat(events, 120)
        data_1 = joined_1.data

        self.assertEqual(joined_1.dt_0, events[0].dt_0)
        self.assertEqual(joined_1.dt_1, events[-1].dt_1)
        self.assertEqual(joined_1.d_t, events[-1].dt_1 - events[0].dt_0)
        self.assertTrue((data_1.index == range(1, data_1.shape[0] + 1)).all())

        # Overlapping events.
        path_files = Path('../test_data/join_files/')
        file_names = ['2020.01.01_00.00.00.txt',
                      '2020.01.01_00.00.10.txt',
                      '2020.01.01_00.00.20.txt',
                      '2020.01.01_00.00.30.txt',
                      '2020.01.01_00.00.40.txt',
                      '2020.01.01_00.00.50.txt',
                      '2020.01.01_00.01.00.txt',
                      '2020.01.01_00.01.10.txt']
        events = []
        for file in file_names:
            ev = event.Event(Path(path_files, file))
            ev.load_data
            events.append(ev)
        joined_2 = event.concat(events, 120)
        data_2 = joined_2.data
        # joined_2.export('../output', 'concat.txt')

        self.assertEqual(joined_2.dt_0, events[0].dt_0)
        self.assertEqual(joined_2.dt_1, events[-1].dt_1)
        self.assertEqual(joined_2.d_t, events[-1].dt_1 - events[0].dt_0)
        self.assertTrue((data_2.index == range(1, data_2.shape[0] + 1)).all())
        self.assertEqual(data_2.shape, (18000, 4))
        self.assertTrue((data_2.index == range(1, 18001)).all())
        self.assertTrue(np.allclose(data_2['t'], np.arange(0, 90, .005)))

    # Summary calculations

    def test_calc_amps(self):
        """Test correct ouptutsize, known values and colum multiindex."""

        # Load info and override settings.
        self.ev_1.load_info('../test_data/event_amp_info.csv')
        self.ev_1.c_stgs.min_amp = 0
        self.ev_1.c_stgs.modes = list(range(1, 6))
        self.ev_1.c_stgs.win_func = None
        # self.ev_1.c_stgs.win_func = 'hann'

        self.ev_1.calc_amps(self.size)
        amps_1 = self.ev_1.amps

        self.ev_1.select_reg(25000, 50001)
        self.ev_1.calc_amps(2**11)
        amps_2 = self.ev_1.amps

        amps_2_mln = [7.988528, 0.737823, 4.681256, 0.8401647, 1.513304]
        amps_2_pyt = amps_2.loc[0, 'c1']

        self.assertEqual(amps_1.shape, (86, 18))
        self.assertTrue(np.allclose(amps_2_mln, amps_2_pyt, rtol=1e-05))
        self.assertIsInstance(amps_1.columns, pd.MultiIndex)

    def test_calc_freqs(self):
        """Test correct ouptut size, known values and colum multiindex."""

        # Load info and override settings.
        self.ev_1.load_info('../test_data/event_amp_info.csv')
        self.ev_1.c_stgs.modes = list(range(1, 6))
        self.ev_1.c_stgs.win_func = None

        self.ev_1.select_reg(25000, 25000 + 2**11 * 2)
        self.ev_1.calc_freqs(2**11)
        freqs_1 = self.ev_1.freqs

        freqs_1_mln = [1.4744, 2.9948, 4.4865, 5.9614, 7.4376]
        freqs_1_pyt = freqs_1.loc[0, 'c1']

        self.assertEqual(freqs_1.shape, (2, 18))
        self.assertTrue(np.allclose(freqs_1_mln, freqs_1_pyt, rtol=1e-03))
        self.assertIsInstance(freqs_1.columns, pd.MultiIndex)


if __name__ == '__main__':
    import logging.config
    logging.config.fileConfig('../log/logging.conf')
    unittest.main()
