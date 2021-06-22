"""Event module to operate with Ml dynamic events.

SuperClass from which to build functionalities
on inherited classes for the different works ahead.
"""

import logging
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import dsp
import settings

logger = logging.getLogger(__name__)

class Event():
    """Class representing a dynamic event. Allows to isolate the
    vibratoin modes through a frequency domain analysis.

    Attributes:

        file_path (str): file name or path to event file.
        fname (str): file name of the event database.
        title (str): title describing the event database.

        data (pd.DataFrame): event loaded and/or modified data.
        info (pd.DataFrame): event information per channel.
        rate (int): measuring rate of the event.

        c_stgs (TYPE): calculation settings
        d_stgs (TYPE): display settings

        rms (TYPE): calculated windowed rms values
        amps (TYPE): calculated windowed main freq amplitude values
        freqs (TYPE): calculated windowed main frequency values

        d_t (datetime): event delta time, duration
        dt_0 (datetime): event start datetime
        dt_1 (datetime): event finish datetime
    """

    def __init__(self, file_path):
        """Initialize an instance with basic attributes.

        Args:
            file_path (str): File name or path.
            freq_main (float): Main vibration frequency of the cable.
        """
        self.file_path = Path(file_path)
        self.fname = self.file_path.name
        self.d_stgs = settings.DisplaySettings()
        self.c_stgs = settings.CalculationSettings()
        logger.info(f'{self} create')

    def __str__(self):
        """Print readable representation of the Event instance"""
        class_name = type(self).__name__
        return f'{class_name}({self.fname})'

    def __repr__(self):
        """Return unambiguous representation of Event instance"""
        class_name = type(self).__name__
        return f'{class_name}({self.file_path})'

    # Data input output

    def load_data(self, dropna=False):
        """Load Ml dynamic database.

        Args:
            dropna (bool, optional): delete empty channels from db.

        Raises:
            FileNotFoundError: if file no found.
            ValueError: if wrong data on file.
        """
        # Load data, delete Ml index, get number of channels, add
        df = pd.read_csv(self.file_path, header=None, index_col=0, dtype='float64')

        cols = df.shape[1]
        if cols < 2:
            raise ValueError(f'{self} wrong file type.')

        df.columns = ['t'] + [f"c{i}" for i in range(1, cols)]
        df.index = df.index.astype(int)
        df.index.name = 'r'

        if dropna:
            df.dropna(axis=1, how='all', inplace=True)

        self.set_data(df)

    def load_info(self, file_path):
        """Set event information as attribure. Must be a csv file
        containing channels as index and event information in columns.

        Args:
            file_path (str): file path.

        """
        info = pd.read_csv(file_path, header=0, index_col=0)
        self.set_info(info)
        logger.info(f'{self} load info')

    def export(self, directory, fname=None):
        """Export data as Ml dynamic database."""
        if not fname:
            try:
                fname = self.dt_0.strftime('%Y.%m.%d_%H.%M.%S') + '.txt'
            except AttributeError:
                fname = self.fname

        self.data.to_csv(Path(directory, fname), header=None, float_format='%.7g')
        logger.info(f'{self} export to folder {directory}.')

    # Setters

    def set_data(self, df):
        """Set event data as attribute.

        Args:
            df (pd.DataFrame): Ml event format df.

        Raises:
            ValueError: if wrong format df passed.
        """
        # Check data is correct.
        cols = df.shape[1]
        conditions = [cols > 2,
                      df.index.name == 'r',
                      df.columns[0] == 't']
        if False in conditions:
            raise ValueError(f'{self} wrong data set.')

        # Set attributes and log
        self.data = df
        self._set_rate()
        logger.debug(f'{self} set data')

    def _set_rate(self):
        """Set as attribure the measuring rate of the event."""
        interval = self.data.iloc[2, 0] - self.data.iloc[1, 0]
        self.rate = int(1 / interval)

    def set_info(self, info):
        """Set event information passed in argument as attribute.

        Must contain channels as index and information in columns.
        Supported column format: ['name', 'freq']

        Args:
            info (pd.DataFrame)

        Raises:
            TypeError: if wrong data type is passed to be saved.
            ValueError: if pd.DataFrame has no correct format.
        """

        # Check has channels as index, if not df will raise TypeError.
        cond = [i[0] == 'c' for i in info.index]

        if not np.all(cond):
            raise ValueError(f'{self} wrong info passed.')

        self.info = info

    def set_title(self, title):
        """Set as attribute a title describing the event.

        Args:
            title (str)
        """
        self.title = title

    def set_datetime(self, dt_0=None):
        """Set attributes for start, finish and duration of event data.

        Initial datetime taken from file name or provided in argument.
        If data is cropped by selecting a subset, the subsets start and
        finish dates are saved, not the files global ones.

        Args:
            dt_0 (None, optional): initial datetime to set manually.
        """
        if dt_0:
            self.dt_0 = dt_0
        else:
            # Workaround for suffix containing events.
            datet_string = self.file_path.stem.replace('_j', '')
            split_datet = datet_string.split('_')
            if len(split_datet) > 2:
                datet_string = '_'.join(split_datet[:2])

            # Calculate files initial time and first entry's datetime.
            f_t0 = datetime.datetime.strptime(datet_string, '%Y.%m.%d_%H.%M.%S')
            init_t = datetime.timedelta(seconds=self.data.iloc[0, 0])
            self.dt_0 = f_t0 + init_t

        # Set attributes
        self.d_t = datetime.timedelta(seconds=self.data.iloc[-1, 0])
        self.dt_1 = self.dt_0 + self.d_t

    def set_dt_index(self, dt_0=None):
        """Set as index the datetime of each registry entry in database.

        Args:
            dt_0 (None, optional): initial datetime to set manually.
        """
        self.set_datetime(dt_0)
        dt = pd.to_timedelta(self.data['t'], unit='sec') + self.dt_0
        self.data.index = dt
        self.data.index.name = 'dt'

    # Data selection

    def windows(self, size, overlap=0, rm_offset=False):
        """Split DataFrame into windows, yields DataFrame subsets.

        Args:
            size (int): window size, number of rows.
            overlap (int): window overlap with previous, rows.
            rm_offset (bool, optional): remove offset of output window.

        Raises:
            IndexError: if no window could be made.

        Yields:
            pd.DataFrame: window.

        """
        rows = self.data.shape[0]
        if (0 < size <= rows) and (0 <= overlap < size):
            n = (rows - size) // (size - overlap) + 1

            for i in range(n):
                start = (size - overlap) * i
                end = start + size
                win = self.data.iloc[start:end, :]
                if rm_offset:
                    win_offset = win - win.mean()
                    win_offset['t'] = win['t']
                    yield win_offset

                yield win

        else:
            raise IndexError(f"{self} no possible window of size '{size}'.")

    def select_reg(self, r0, r1, add_zeros=False):
        """Set as attribute a window containing a subset of the data.

        Args:
            r0 (int): first reg entry, inclusive.
            r1 (int): last reg entry, not inclusive.
            add_zeros (bool, optional): add zeros if last_reg > size.

        Raises:
            ValueError: if no window could be made.
        """
        # Retrieve database first and last registry entry values
        r0_db, r1_db = self.data.index[0], self.data.index[-1]

        # Check invalid input, if r1 > r1_db, add zeros or error.
        if r0 < r0_db or r0 >= r1 or r0 > r1_db:
            raise IndexError(f'{self} no possible reg selection'
                             f' for {r0, r1}.')

        elif r1 > r1_db:
            if add_zeros:
                ind_appen = range(r1_db + 1, r1 + 1)
                ad = pd.DataFrame(0, index=ind_appen, columns=self.data.columns)
                self.data = self.data.append(ad)
            else:
                raise IndexError(f'{self} no possible reg selection for '
                                 f'{r0, r1} without adding zeros.')

        # Make the data selection
        self.data = self.data.loc[r0: r1 - 1, :]

    def select_ch(self, channels):
        """Select subset of data with  channels and set as attribute.

        If invalid input (not a list or not in database) log error.

        Args:
            channels (array): Channels to select.

        Raises:
            KeyError: if channel not in database.
            TypeError: if channels has wrong type.

        No Longer Raises:
            ValueError: if invalid channels.

        """
        try:
            chs = ['t'] + list(channels)
            self.data = self.data.loc[:, chs]
        except TypeError:
            raise TypeError(f"{self} channels '{channels}' must be an array.")
        except KeyError:
            raise KeyError(f"{self} channels '{channels}' not in database.")

    def select_vibration(self, delay, ch='c1'):
        """Select the free vibration portion of the event post loading.

        Warns if maximum vibration is not found at beginning of event.

        Args:
            delay (int): seconds after loading to begin selection.
        """
        signal = self.data.loc[:, ch]
        max_ind = np.argmax(signal.abs())
        r0 = max_ind + self.rate * delay
        r1 = self.data.shape[0]

        self.select_reg(r0, r1)

        if r0 > int(0.5 * r1):
            t = round(r0 / self.rate, 2)
            logger.warning(f'{self} dubious loading instant found: t = {t}')

    # Event concatenation

    def overlap(self, other):
        """Calculate the overlapping time between events.

        If no overlap the gap between events is given as negative.

        Args:
            other (Event): second event

        Returns:
            float: overlapping seconds
        """
        self.set_datetime()
        other.set_datetime()
        return (self.dt_1 - other.dt_0).total_seconds()

    def join(self, other, max_gap=3600):
        logger.info(f'{self}, {other} join')
        """Join event databases in a new Event instance.

        Overlapping part due to pre-event data will be removed.
        If gap between events is exceedes max_gab no joinin will occur.
        Both events must have same daq settings (channels, rate).

        Args:
            other (Event): second Event
            max_gap (int, optional): no-data seconds allowed between db

        Returns:
            Event: Event with joined database.

        Raises:
            ValueError: if time between events exceedes max_gap.
        """
        # Get data, trim second event and join.
        overlap = self.overlap(other)
        data_0 = self.data
        data_1 = other.get_data()

        delta_t_events = self.d_t.total_seconds()
        data_1['t'] = data_1['t'] + delta_t_events - overlap
        if overlap > 0:
            # If events overlap trim duplicates on second event.
            ind_overlap = int(self.rate * overlap) + 2
            data_1 = data_1.iloc[ind_overlap:, :]
        elif overlap < - max_gap:
            print(overlap, max_gap)
            raise ValueError(f'{self} and {other} exceed gap: {-int(overlap)} s.')

        joined_data = data_0.append(data_1)
        joined_data.index = range(1, len(joined_data.index) + 1)
        joined_data.index.name = 'r'

        # Create new Event object and load attributes.
        file_name = self.file_path.stem
        if '_j' not in file_name:
            file_name += '_j'
        result = Event(file_name + '.txt')
        result.set_data(joined_data)
        result.set_datetime(self.dt_0)
        return result

    # Summary calculations

    def calc_rms(self, size, overlap=0):
        """Calculates the rms of the event in window sizes

        Data must previously be loaded to the event.
        Sets the rms values for each window and channel as attribure.
        Time is calculated as the central point of the window.

        Args:
            size (int): window size to divide event.
            overlap (int, optional): window overlap. Default is 0.

        """
        cols = ['t', 'r0', 'r1'] + list(self.data.columns[1:])
        df = pd.DataFrame(columns=cols)
        wins = self.windows(size, overlap)

        for row, win in enumerate(wins):
            t, r0, r1 = win.iloc[size // 2, 0], win.index[0], win.index[-1]
            t_row = pd.Series({'t': t, 'r0': r0, 'r1': r1})

            rms_data = win.iloc[:, 1:].apply(dsp.rms, axis=0, samples=size)
            rms_row = t_row.append(rms_data)
            rms_row.name = row

            df = df.append(rms_row)

        self.rms = df.astype({'r0': 'int32', 'r1': 'int32'})
        logger.info(f'{self} calculate rms values')

    def _calc_winmodes(self, size, overlap, func):
        """Calculate passed function on window and mode split database.

        Applies 'func' to each window of the data attribute. Stores
        results on a MultiIndex DataFrame with reg and time information
        folowed by the results for each mode. Supported functions
        from dsp module [find_freq, find_amp].

        Loads CalculationSettings from 'settings.py' file.

        Args:
            size (int): window size to divide event.
            overlap (int, optional): window overlap. Default is 0.
            func (function): function to compute on each window.

        Returns:
            pd.DataFrame: win info & func values for window and mode.
        """
        # Import CalculationSettings to be used as kwargs.
        calc_kwargs = vars(self.c_stgs)

        # Data to use on loops.
        bins = dsp.fft_freq(size, self.rate)
        r_index = pd.MultiIndex.from_product([['r'], ['t', 'r0', 'r1']])
        tuples_ind = (('t', 't'), ('r', 'r0'), ('r', 'r1'))
        r_index = pd.MultiIndex.from_tuples(tuples_ind)
        result = pd.DataFrame()

        # On each window, fill a series with all channel values.
        for n, win in enumerate(self.windows(size, overlap)):
            t, r0, r1 = win.iloc[size // 2, 0], win.index[0], win.index[-1]
            row = pd.Series([t, r0, r1], index=r_index, dtype='float32').round(2)

            # Calculate amplitudes for each channel, add to series.
            for ch in win.columns[1:]:
                main_freq = self.info.loc[ch, 'freq']
                vals = func(win[ch], bins, self.rate, main_freq,
                            samples=size, **calc_kwargs)

                vals.index = pd.MultiIndex.from_product([[ch], vals.index])
                row = row.append(vals)

            # Add series to DataFrame.
            row.name = n
            result = result.append(row)

        # Set MultiIndex as columns. Append does not insert a multi.
        result.columns = row.index
        return result.astype({('r', 'r0'): int, ('r', 'r1'): int})

    def calc_amps(self, size, overlap=0):
        """Sets attribute amplitude of each vibration mode per window.

        Loads CalculationSettings from 'settings.py' file.
        Must load data and set info attribute beforehand.

        Args:
            size (int): Number of entries of each window.
            overlap (int, optional): window overlap. Default is 0.

        Returns:
            pd.DataFrame: window info and amps for mode and channel.

        """
        df = self._calc_winmodes(size, overlap, dsp.find_amps)
        self.amps = df
        logger.info(f'{self} calculate amp values')

    def calc_freqs(self, size, overlap=0):
        """Sets attribute frequencies of each vibration mode per window.

        Loads CalculationSettings from 'settings.py' file.
        Must load data and set info attribute beforehand.

        Args:
            size (int): Number of entries of each window.
            overlap (int, optional): window overlap. Default is 0.

        Returns:
            pd.DataFrame: window info and freqs for mode and channel.

        """
        df = self._calc_winmodes(size, overlap, dsp.find_freqs)
        self.freqs = df
        logger.info(f'{self} calculate freq values')

    # Getters
    def get_data(self):
        """Retrieve event data copy.

        For use when data might be modified.

        Returns:
            pd.DataFrame: event data
        """
        return self.data.copy()

    def get_title(self):
        """Retrieve tittle attribute

        Returns:
            str: event title
        """
        return self.title

    def get_rms(self):
        """Return the calculated rms values of the event.

        Returns:
            pd.DataFrame: info and rms values for each channel.
        """
        return self.rms.copy()

    def get_amps(self):
        """Return the calculated amplitudes values of the event.

        Returns:
            pd.DataFrame: info and amplitudes for each mode and channel.
        """
        return self.df_fix(self.amps.copy(), 'amps')

        return df

    def get_freqs(self):
        """Return the calculated main frequency values of the event.

        Returns:
            pd.DataFrame: info and main freq for each mode and channel.
        """
        return self.df_fix(self.freqs.copy(), 'freqs')

    # Formatters

    def df_fix(self, df, units='freqs'):
        """Set fix (decimal) values to a results MultiIndex DataFrame.

        Sets fix of window information (time and registry) and values.
        Loads DisplaySettings from 'settings.py' file.
        Supported units ['amps', 'freqs'].

        Args:
            df (pd.DataFrame): results MultiIndex df to set fix.
            units (str, optional): values units.

        Returns:
            pd.DataFrame: DataFrame with set fix values.

        Raises:
            ValueError: if wrong units are passed
        """

        # Import display settings.
        disp_stgs = vars(self.d_stgs)

        if units == 'freqs':
            vals_fix = disp_stgs['freq_fix']
        elif units == 'amps':
            vals_fix = disp_stgs['amps_fix']
        else:
            raise ValueError(f"invalid DataFrame type '{units}'.")

        df['t', 't'] = df['t', 't'].round(disp_stgs['time_fix'])
        df.iloc[:, 3:] = df.iloc[:, 3:].round(vals_fix)

        return df

    # Developing methods

# General functions

def concat(events, max_gap=3600):
    """Concatenate event databases and return new event object.

    Args:
        events (list): events sequence to be joined.
        max_gap (int, optional): no-data seconds allowed between db.

        Returns:
            Event: Event with joined databases.

        Raises:
            ValueError: if time between events exceedes max_gap.
    """
    # [i.load_data() for i in events]

    result = events[0]
    result.load_data()

    for i in range(1, len(events)):
        events[i].load_data()
        result = result.join(events[i], max_gap)

    return result


# def


def main():
    a = 8
    print("Nothing done Jone")


if __name__ == '__main__':

    # If module directly run, load log configuration for all modules.
    import logging.config
    logging.config.fileConfig('../log/logging.conf')
    logger = logging.getLogger('event')

    main()
