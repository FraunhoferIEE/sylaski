
from typing import List, Tuple, Iterator, Optional

import pandas as pd
import numpy as np


def shift_array(arr, num, fill_value=None):
    """Shift numpy array by num values."""

    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


class TimeSeriesSplitter:
    """Split dataframe with time series in shorter time series, taking data gaps into account,
    so only complete windows are selected.

    Args:
        window_length: length of the windows to generate in hours. Default 24.
        overlap: the overlap of the windows in hours. Default 0 (no overlap).
        ts_freq: the frequency of the time series in minutes. Default 10.
            This is used to determine where there are data gaps.
    """

    def __init__(self, window_length: int = 24, overlap: int = 0, ts_freq: int = 10):
        self.window_length: int = window_length
        self.overlap: int = overlap

        self.freq: int = ts_freq

        self.windows: Optional[List[Tuple[np.datetime64, np.datetime64]]] = None
        self.data_gaps: Optional[np.array] = None

    def split_time_series(self, df: pd.DataFrame) -> Iterator[np.array]:
        """Split time series dataframe. Output is a generator.

        Args:
            df: pandas DataFrame with time series. The timestamp is expected to be the index (a pandas DatetimeIndex).

        Returns:
            Generator creating arrays of specified window length.
        """
        data_gaps = self._get_data_gap_starts(timestamps=df.index.to_series())
        windows = self._get_ts_windows(start=df.index.min().to_numpy(),
                                       end=df.index.max().to_numpy(),
                                       data_gaps=data_gaps)

        data = df.values
        timestamps = df.index.values
        for window in windows:
            yield data[np.where(timestamps == window[0])[0][0]:np.where(timestamps == window[1])[0][0]]

    def _get_data_gap_starts(self, timestamps: np.array) -> np.array:
        """Get the start of the data gaps in the given time series as a numpy array."""
        starts = timestamps[shift_array(timestamps, -1) - timestamps > np.timedelta64(self.freq, 'm')]
        ends = timestamps[timestamps - shift_array(timestamps, 1) > np.timedelta64(self.freq, 'm')]
        self.data_gaps = np.array(list(zip(starts, ends)))
        return self.data_gaps

    def _get_ts_windows(self, start: np.datetime64, end: np.datetime64, data_gaps: np.array) -> List[Tuple[np.datetime64, np.datetime64]]:
        """Generate a list of possible time windows as a list of (start, end) tuples.

        Args:
            start: start of the time series
            end: end of the time series
            data_gaps: list or numpy array with timestamps that indicate start of a data gap.

        Returns:
            list of (start, end) tuples.
        """

        windows = []

        if len(data_gaps) == 0:
            # If there are no data gaps
            windows = self._create_windows(start=start, end=end)
            self.windows = windows
            return windows

        # up to and between data gaps
        window_start = start
        data_gap_end = None
        for data_gap_start, data_gap_end in data_gaps:
            windows.extend(self._create_windows(start=window_start, end=data_gap_start))
            window_start = data_gap_end

        # Before end of time series, after last data gap
        windows.extend(self._create_windows(start=data_gap_end, end=end))

        self.windows = windows
        return windows

    def _create_windows(self, start: np.datetime64, end: np.datetime64) -> List[Tuple[np.datetime64, np.datetime64]]:
        """Create windows between given start and end datetime."""
        windows = []
        window_start = start

        if end - start >= np.timedelta64(self.window_length, 'h'):
            while True:
                window_end = window_start + np.timedelta64(self.window_length, 'h')
                if window_end > end:
                    break
                windows.append((window_start, window_end))
                window_start = window_end - np.timedelta64(self.overlap, 'h')
        return windows
