""" Settings file to  edit event calculations

Handles min & max default values as well as displaying options
"""

class CalculationSettings():
    """Settings class for Event objects. Attributes contain values
    to be used during the computation.

    Attributes:
        min_amp (int): minimum amplitude threshold to store values.
        modes (int): number of modes (multiples) to split window.
        width (float): % of the freq_mode wher to search for  amplitude.
        win_func (str): windowing function to be applied to the signal.
    """

    def __init__(self):
        """Instanciation function to enter settings manually."""

        # Window and mode spliting
        self.width = .2
        self.modes = [1, 2, 3, 4]

        # Signal windowing.
        self.win_func = 'hann'
        # Amplitude calculation.

        self.min_amp = 0

        # self.adj_cent = True

class DisplaySettings(object):
    """Event objects settings. Attributes contain computation kwargs.

    Attributes:
        amps_fix (int): time column number of decimals.
        freq_fix (int): freq values number of decimals.
        time_fix (int): amp values number of decimals.

    """

    def __init__(self):
        """Instanciation function to enter settings manually."""

        self.time_fix = 2
        self.amps_fix = 2
        self.freq_fix = 4


def main():
    pass


if __name__ == '__main__':
    main()
