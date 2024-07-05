## averaging data to average winter/summer/transition week based on VDEW (1999) and extraction of features
# (https://www.bdew.de/media/documents/1999_Repraesentative-VDEW-Lastprofile.pdf):
#   winter:     first Monday in December and the following 3 weeks
#               first Monday after New Year and the following 8 weeks
#   summer:     first Monday in June and the following 16 weeks
#   transition: first Monday in April and the following 8 weeks
#               first Monday in October and the following 4 weeks
#
## NOTE:    input must be a DataFrame (columns = different samples (profiles), rows = time steps of one profile)
#           with datetime index

## import modules
import pandas as pd
import numpy as np
import warnings
import time
from datetime import datetime, timedelta
from scipy import stats
from itertools import chain
import copy
from sklearn.preprocessing import MinMaxScaler
import scipy.signal as signal


## function to find the first Monday of a month
def first_monday(year, month):
    """
    Find the date of the first Monday in a given year and month.

    Parameters:
    -----------
    year : int
        The year for which to find the first Monday.
    month : int
        The month for which to find the first Monday.

    Returns:
    --------
    pandas.Timestamp
        The date of the first Monday in the specified year and month.

    Notes:
    ------
    This function calculates the date of the first Monday in the given year and month.
    """
    # getting date of first Monday
    if month < 10:
        year_month = str(year) + '-0' + str(month)
    else:
        year_month = str(year) + '-' + str(month)
    date = np.busday_offset(year_month, 0,
                            roll='forward',
                            weekmask='Mon')

    return pd.to_datetime(date)


## function to change time resolution of aggregated profiles to 15 minute steps
def timestep15(df):
    """
    Change the time resolution of aggregated profiles to 15-minute steps.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the aggregated profiles.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with time resolution changed to 15-minute steps.

    Notes:
    ------
    This function resamples the input DataFrame to aggregate the data to 15-minute time steps.
    """

    df_agg15 = copy.deepcopy(df)
    sec_per_day = int(3600 * 24 * 7 * 3)  # 3600s x 24h x 7 days x 3 weeks
    sec = int(sec_per_day / len(df_agg15[df_agg15.columns[0]]))  # get resolution [s] of current data
    # sec = (df.index[1]-df.index[0]).seconds  # get resolution [s] of current data

    # placeholder date for resampling
    date_type = 'datetime64[' + str(sec) + 's]'  # set time steps
    date_range = pd.to_datetime(np.arange('2000-01-01 00:00:00', '2000-01-22 00:00:00', dtype=date_type))

    # aggregate to 15 minute steps
    if sec < 900:
        df_agg15.index = date_range
        df_agg15 = df_agg15.resample('900s').mean()
        df_agg15.index = range(len(df_agg15))
    else:
        df_agg15 = pd.DataFrame(columns=df.columns)
        for c in df_agg15.columns:
            df_agg15[c] = df[c].repeat(sec/900)
        df_agg15.index = pd.to_datetime(np.arange('2000-01-01 00:00:00', '2000-01-22 00:00:00',
                                                  dtype='datetime64[900s]'))

    return df_agg15


## function to create date range for winter/summer/transition days according to VDEW (1999)
def seasons_vdew(df):
    """
    Create date ranges for winter, summer, and transition days according to VDEW (1999) guidelines.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.

    Returns:
    --------
    list
        A list containing three pandas.DatetimeIndex objects representing winter, summer, and transition days.

    Notes:
    ------
    This function generates date ranges for winter, summer, and transition days based on the VDEW (1999) guidelines.
    """

    # get occurring years in data frame
    years = np.unique(df.index.year)

    # placeholders
    winter = []
    summer = []
    transition = []

    # get date range of winter, summer and transition days
    for y in years:
        winter.extend(pd.date_range(first_monday(y, 1), first_monday(y, 1) + timedelta(weeks=8)).append(
            pd.date_range(first_monday(y, 12), first_monday(y, 12) + timedelta(weeks=3))))
        summer.extend(pd.date_range(first_monday(y, 6), first_monday(y, 6) + timedelta(weeks=16)))
        transition.extend(pd.date_range(first_monday(y, 4), first_monday(y, 4) + timedelta(weeks=8)).append(
            pd.date_range(first_monday(y, 10), first_monday(y, 10) + timedelta(weeks=4))))

    # convert to datetime
    winter = pd.to_datetime(winter)
    summer = pd.to_datetime(summer)
    transition = pd.to_datetime(transition)

    return [winter, summer, transition]


## function to create aggregated normalized load profiles for average weeks
def average_weeks(df):
    """
    Create aggregated normalized load profiles for average weeks.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the load profile data.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the aggregated normalized load profiles for average weeks, categorized by season and day of the week.

    Notes:
    ------
    This function aggregates the load profile data into average weeks, normalized by their maximum value.
    It categorizes the data by season (winter, summer, transition) and day of the week (Monday to Sunday).
    """

    df_copy = copy.deepcopy(df)
    # normalize data with maximum value
    df_copy = df_copy / df_copy.max()

    # create lists for aggregation
    season_list = seasons_vdew(df_copy)
    season_names = ['winter', 'summer', 'transition']
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # aggregated profiles (average weeks)
    df_season_week = pd.DataFrame()  # placeholder
    for s in range(len(season_names)):
        # get data for specific season
        df_season = df_copy[np.in1d(pd.to_datetime(df_copy.index.date), season_list[s])]
        df_weekday_agg = pd.DataFrame()  # placeholder
        for day in weekdays:
            # aggregate data of specific season to one average week
            df_weekday = df_season[df_season.index.day_name() == day]
            df_weekday_agg = pd.concat(
                [df_weekday_agg, df_weekday.groupby(df_weekday.index.time).mean()], ignore_index=True)

        # data frame with all 3 weeks (winter, summer, transition)
        df_season_week = pd.concat([df_season_week, df_weekday_agg])

    # set multi index (winter, summer, transition)
    df_season_week.index = pd.MultiIndex.from_product([season_names, range(len(df_weekday_agg))])

    return df_season_week


## function to calculate daily parameters to describe load profile (features)
def get_param(df):
    """
    Calculates various statistical parameters from the input DataFrame `df`.

    Parameters:
    - df: pandas DataFrame
        The input DataFrame containing the load profile data.

    Returns:
    - df_param: pandas DataFrame
        A DataFrame containing the calculated statistical parameters.

    Statistical Parameters:
    1. Mean: Daily mean value of the load profile.
    2. Standard Deviation: Daily standard deviation of the load profile.
    3. Morning: Ratio of the mean values during morning hours (05:00-11:00) to the overall mean.
    4. Day: Ratio of the mean values during daytime hours (10:00-16:00) to the overall mean.
    5. Evening: Ratio of the mean values during evening hours (16:00-23:00) to the overall mean.
    6. Night: Daily mean value during nighttime hours (00:00-06:00).
    7. Skewness: Daily skewness of the load profile.
    8. Quantile 25: Daily 25th percentile of the load profile.
    9. Quantile Distance: Daily difference between the 75th and 25th percentiles of the load profile.
    10. Load Factor: Daily load factor of the load profile.
    11. Coincidence Factor: Daily coincidence factor of the load profile.
    12. Peak-Valley Ratio: Daily ratio of the difference between the peak and valley values to the valley value.
    13. Peak Load Factor: Daily peak load factor of the load profile.
    14. Flat Load Factor: Daily flat load factor of the load profile.
    15. Valley Load Factor: Daily valley load factor of the load profile.
    16. Non-linear Metric: Daily non-linear metric of the load profile.
    17. Linear Regression: Regression coefficient between neighboring days.
    18. Peak Number: Daily number of peaks in the load profile.
    19. Peak Location: Hourly location of peaks in the load profile.
    20. Peak Magnitude: Hourly magnitude of peaks in the load profile.
    21. Peak Width: Hourly width of peaks in the load profile.

    Notes:
    - The function employs resampling to calculate daily parameters from the input load profile data.
    - It also handles cases where there might not be enough data available for certain calculations.
    """

    warnings.filterwarnings("ignore")
    df_copy = copy.deepcopy(df)

    if isinstance(df_copy.index[0], datetime):
        # get first occurring year in data frame as string
        year_str = str(df_copy.index.year[0])
        current_step = (df_copy.index[1] - df_copy.index[0]).seconds
    else:
        # if DataFrame is already aggregated
        # current_step = int(86400 / (len(df_copy) / 12))  # seconds per day / timestamps per day
        # example_date = pd.to_datetime(np.arange('2000-01-01 00:00:00', '2000-01-13 00:00:00',
        #                                         dtype='datetime64[' + str(current_step) + 's]'))
        current_step = int(86400 / (len(df_copy) / 21))  # seconds per day / timestamps per day
        example_date = pd.to_datetime(np.arange('2000-01-01 00:00:00', '2000-01-22 00:00:00',
                                                dtype='datetime64[' + str(current_step) + 's]'))
        df_copy.index = example_date
        year_str = str(df_copy.index.year[0])

    # define functions to calculate specific daily parameters
    def skew(data):
        data = data[~pd.isnull(data)]  # remove NaNs
        # skewness
        v = 0  # placeholder
        for i in data:
            v = ((i - np.mean(data)) / np.std(data))**3 + v
        return v/len(data)

    def morning(data):
        # morning time (05:00-11:00)
        idx1 = df_copy.index.get_loc(year_str + '-01-01 05:00:00')
        idx2 = df_copy.index.get_loc(year_str + '-01-01 11:00:00')
        morning_time = range(idx1, idx2)
        if any(len(data) < i for i in morning_time):
            return np.nan
        else:
            return data.iloc[morning_time].mean() / data.mean()

    def evening(data):
        # evening time (13:00-19:00)
        idx1 = df_copy.index.get_loc(year_str + '-01-01 13:00:00')
        idx2 = df_copy.index.get_loc(year_str + '-01-01 19:00:00')
        evening_time = range(idx1, idx2)
        if any(len(data) < i for i in evening_time):
            return np.nan
        else:
            return data.iloc[evening_time].mean() / data.mean()

    def av_night(data):
        # average value of nighttime period
        idx1 = df_copy.index.get_loc(year_str + '-01-01 00:00:00')
        idx2 = df_copy.index.get_loc(year_str + '-01-01 06:00:00')
        night_time = range(idx1, idx2)
        if any(len(data) < i for i in night_time):
            return np.nan
        else:
            return data.iloc[night_time].mean()

    def av_morning(data):
        # average value of morning period
        idx1 = df_copy.index.get_loc(year_str + '-01-01 06:00:00')
        idx2 = df_copy.index.get_loc(year_str + '-01-01 10:00:00')
        morning_time = range(idx1, idx2)
        if any(len(data) < i for i in morning_time):
            return np.nan
        else:
            return data.iloc[morning_time].mean()

    def av_day(data):
        # average value of daytime period
        idx1 = df_copy.index.get_loc(year_str + '-01-01 10:00:00')
        idx2 = df_copy.index.get_loc(year_str + '-01-01 16:00:00')
        day_time = range(idx1, idx2)
        if any(len(data) < i for i in day_time):
            return np.nan
        else:
            return data.iloc[day_time].mean()

    def av_evening(data):
        # average value of evening period
        idx1 = df_copy.index.get_loc(year_str + '-01-01 16:00:00')
        idx2 = df_copy.index.get_loc(year_str + '-01-01 23:00:00')
        evening_time = range(idx1, idx2)
        if any(len(data) < i for i in evening_time):
            return np.nan
        else:
            return data.iloc[evening_time].mean()

    def std_night(data):
        # average value of nighttime period
        idx1 = df_copy.index.get_loc(year_str + '-01-01 00:00:00')
        idx2 = df_copy.index.get_loc(year_str + '-01-01 06:00:00')
        night_time = range(idx1, idx2)
        if any(len(data) < i for i in night_time):
            return np.nan
        else:
            return data.iloc[night_time].std()

    def std_morning(data):
        # average value of morning period
        idx1 = df_copy.index.get_loc(year_str + '-01-01 06:00:00')
        idx2 = df_copy.index.get_loc(year_str + '-01-01 10:00:00')
        morning_time = range(idx1, idx2)
        if any(len(data) < i for i in morning_time):
            return np.nan
        else:
            return data.iloc[morning_time].std()

    def std_day(data):
        # average value of daytime period
        idx1 = df_copy.index.get_loc(year_str + '-01-01 10:00:00')
        idx2 = df_copy.index.get_loc(year_str + '-01-01 16:00:00')
        day_time = range(idx1, idx2)
        if any(len(data) < i for i in day_time):
            return np.nan
        else:
            return data.iloc[day_time].std()

    def std_evening(data):
        # average value of evening period
        idx1 = df_copy.index.get_loc(year_str + '-01-01 16:00:00')
        idx2 = df_copy.index.get_loc(year_str + '-01-01 23:00:00')
        evening_time = range(idx1, idx2)
        if any(len(data) < i for i in evening_time):
            return np.nan
        else:
            return data.iloc[evening_time].std()

    def pLoadFactor(data):
        # peak load factor (08:00-11:00, 18:00-21:00)
        idx1 = df_copy.index.get_loc(year_str + '-01-01 08:00:00')
        idx2 = df_copy.index.get_loc(year_str + '-01-01 11:00:00')
        idx3 = df_copy.index.get_loc(year_str + '-01-01 18:00:00')
        idx4 = df_copy.index.get_loc(year_str + '-01-01 21:00:00')
        l = [idx1, idx2, idx3, idx4]
        peak_time = chain(range(idx1, idx2), range(idx3, idx4))
        if max(l) > len(data):
            return np.nan
        else:
            return data.iloc[peak_time].mean() / data.mean()

    def fLoadFactor(data):
        # flat load factor (06:00-08:00, 11:00-18:00, 21:00-22:00)
        idx1 = df_copy.index.get_loc(year_str + '-01-01 06:00:00')
        idx2 = df_copy.index.get_loc(year_str + '-01-01 08:00:00')
        idx3 = df_copy.index.get_loc(year_str + '-01-01 11:00:00')
        idx4 = df_copy.index.get_loc(year_str + '-01-01 18:00:00')
        idx5 = df_copy.index.get_loc(year_str + '-01-01 21:00:00')
        idx6 = df_copy.index.get_loc(year_str + '-01-01 22:00:00')
        l = [idx1, idx2, idx3, idx4, idx5, idx6]
        flat_time = chain(range(idx1, idx2), range(idx3, idx4), range(idx5, idx6))
        if max(l) > len(data):
            return np.nan
        else:
            return data.iloc[flat_time].mean() / data.mean()

    def vLoadFactor(data):
        minute = str(int(current_step / 60))
        # valley load factor (22:00-24:00, 00:00-06:00)
        idx1 = df_copy.index.get_loc(year_str + '-01-01 22:00:00')
        idx2 = df_copy.index.get_loc(year_str + '-01-02 00:00:00')
        idx3 = df_copy.index.get_loc(year_str + '-01-01 00:'+minute+':00')
        idx4 = df_copy.index.get_loc(year_str + '-01-01 06:00:00')
        l = [idx1, idx2, idx3, idx4]
        valley_time = chain(range(idx1, idx2), range(idx3, idx4))
        if max(l) > len(data):
            return np.nan
        else:
            return data.iloc[valley_time].mean() / data.mean()

    def nonLinear(data):
        # non-linear metrics
        if len(data) < 3:
            return np.nan
        else:
            return np.sum((data.iloc[2]) ** 2 * data.iloc[1:-1] * data.iloc[:-2], axis=0) / len(data - 2)

    def meanChange(data):
        data = data[~pd.isnull(data)]  # remove NaNs
        # mean absolute change
        return np.mean(abs(np.diff(data)))

    def peak_no(data):
        # function of number of occurring peaks
        data = data.resample('h').mean()
        p = signal.find_peaks(data, height=data.mean())
        loc = np.zeros(24)
        loc[p[0]] = 1
        num = sum(loc)
        return num

    def peak_loc(data):
        # function of location of occurring peaks
        data = data.resample('h').mean()
        p = signal.find_peaks(data, height=data.mean())
        loc = np.zeros(24)
        loc[p[0]] = 1
        return [loc]

    def peak_mag(data):
        # function of magnitude of occurring peaks
        data = data.resample('h').mean()
        p = signal.find_peaks(data, height=data.mean())
        y = data[p[0]]
        mag = np.zeros(24)
        mag[p[0]] = y
        return [mag]

    def peak_width(data):
        # function of magnitude of occurring peaks
        data = data.resample('h').mean()
        p = signal.find_peaks(data, height=data.mean())
        w = signal.peak_widths(data, p[0])
        width = np.zeros(24)
        width[p[0]] = w[0]
        return [width]

    def linear_reg(dframe):
        df_reg = pd.DataFrame()
        for profile in dframe.columns:
            df_h = dframe[profile].resample('h').mean()  # aggregate to hourly resolution
            indx = [df_h.index.date, df_h.index.time]
            df_h = pd.DataFrame(df_h)
            df_d = df_h.set_index(indx).unstack(level=0)  # rearrange -> column for every day
            r = np.empty(len(df_d.columns))
            r.fill(np.nan)
            for day in range(len(df_d.columns)):
                if day == len(df_d.columns) - 1:
                    next_day = 0  # compare last day with first day of data (makes only sense for averaged weeks)
                else:
                    next_day = day + 1
                mask = ~np.isnan(df_d.iloc[:, day]) & ~np.isnan(df_d.iloc[:, next_day])
                # prevent linear regression with empty vector (only NaNs) or identical values
                if mask.any() & (len(np.unique(df_d.iloc[:, day][mask])) > 1):
                    slope, intercept, r[day], p, std_err = stats.linregress(
                        df_d.iloc[:, day][mask], df_d.iloc[:, next_day][mask])
                else:
                    r[day] = np.nan
            df_reg[profile] = r
            df_reg.index = dframe.resample('D').mean().index
        return df_reg

    # create columns in data frame with calculated daily parameters
    st = time.time()

    features = ['mean', 'std', 'avMorning', 'avDay', 'avEvening', 'avNight', 'stdMorning', 'stdDay', 'stdEvening',
                'stdNight', 'skew', 'quan25', 'quanDist', 'loadFactor', 'coincidenceFactor', 'peakValley',
                'pLoadFactor', 'fLoadFactor', 'vLoadFactor', 'morning', 'evening', 'meanChange', 'nonLinear', 'linReg',
                'peakNo', 'peakLoc', 'peakMag', 'peakWidth']

    # make sure that calculations are concatenated in the same order as they appear in "features" vector above
    # otherwise the multi index added below will be wrong
    df_param1 = pd.concat([
        df_copy.resample('D').mean(),  # daily mean
        df_copy.resample('D').std(),  # daily standard deviation
        df_copy.resample('D').apply(av_morning),  # daily mean of morning values
        df_copy.resample('D').apply(av_day),  # daily mean of daytime values
        df_copy.resample('D').apply(av_evening),  # daily mean of evening values
        df_copy.resample('D').apply(av_night),  # daily mean of nighttime values
        df_copy.resample('D').apply(std_morning),  # daily standard deviation of morning values
        df_copy.resample('D').apply(std_day),  # daily standard deviation of daytime values
        df_copy.resample('D').apply(std_evening),  # daily standard deviation of evening values
        df_copy.resample('D').apply(std_night),  # daily standard deviation of nighttime values
        df_copy.resample('D').apply(skew),  # daily skewness
        #df_copy.resample('D').apply(stats.skew),
        df_copy.resample('D').quantile(0.25),  # daily 25 percent quantile
        df_copy.resample('D').quantile(0.75) - df_copy.resample('D').quantile(0.25),  # daily quantile distance
        df_copy.resample('D').mean() / df_copy.resample('D').max(),  # daily load factor
        df_copy.resample('D').max() / df_copy.max(),  # daily coincidence factor
        (df_copy.resample('D').max() - df_copy.resample('D').min()) / df_copy.resample('D').min(),
        # daily peak-valley ratio
        df_copy.resample('D').apply(pLoadFactor),  # daily peak load factor
        df_copy.resample('D').apply(fLoadFactor),  # daily flat load factor
        df_copy.resample('D').apply(vLoadFactor),  # daily valley load factor
        df_copy.resample('D').apply(morning),  # daily ratio of morning values to average
        df_copy.resample('D').apply(evening),  # daily ratio of evening values to average
        df_copy.resample('D').apply(meanChange),  # daily mean absolute change
        df_copy.resample('D').apply(nonLinear),  # daily non-linear metric
        linear_reg(df_copy),  # regression coefficient between neighboring days
        df_copy.resample('D').apply(peak_no),  # daily number of peaks
    ])

    # remove inf
    for c in df_copy.columns:
        df_param1[c][df_param1[c] == float('inf')] = 0

    # add multi index for better overview
    # index for daily parameters
    day_no = len(df_copy.resample('D').mean())
    index1 = pd.MultiIndex.from_product([features[:-3], range(day_no)])
    df_param1.index = index1

    # peak functions
    loc = df_copy.resample('D').apply(peak_loc)  # daily location of peaks (logical)
    mag = df_copy.resample('D').apply(peak_mag)  # daily magnitude of peaks
    width = df_copy.resample('D').apply(peak_width)
    df_loc = pd.DataFrame()
    df_mag = pd.DataFrame()
    df_width = pd.DataFrame()
    for i in df_copy.columns:
        temp_loc = loc[i].explode().explode()
        temp_mag = mag[i].explode().explode()
        temp_width = width[i].explode().explode()
        df_loc[i] = temp_loc
        df_mag[i] = temp_mag
        df_width[i] = temp_width
    df_param2 = pd.concat([df_loc, df_mag, df_width])

    # index for hourly parameters (location, magnitude and width of peaks)
    hour_no = len(df_copy.resample('h').mean())
    arrays = [np.concatenate([[features[-3]] * hour_no, [features[-2]] * hour_no, [features[-1]] * hour_no]),
              np.concatenate([range(hour_no), range(hour_no), range(hour_no)])]
    tuples = list(zip(*arrays))
    index2 = pd.MultiIndex.from_tuples(tuples)
    df_param2.index = index2

    # concatenate daily and hourly features
    df_param = pd.concat([df_param1, df_param2])

    et = time.time()
    print("Done! Execution time:", round((et - st) / 60, ndigits=2), "minutes")

    return df_param


## function to normalize features
def norm_param(df_param):
    """
    Normalizes the features in the input DataFrame `df_param`.

    Parameters:
    - df_param: pandas DataFrame
        The input DataFrame containing the statistical parameters to be normalized.

    Returns:
    - df_n_feature: pandas DataFrame
        A DataFrame containing the normalized features.

    Features:
    - mean: Daily mean value of the load profile.
    - std: Daily standard deviation of the load profile.
    - avMorning: Daily mean of morning values.
    - avDay: Daily mean of daytime values.
    - avEvening: Daily mean of evening values.
    - avNight: Daily mean of nighttime values.
    - stdMorning: Daily standard deviation of morning values.
    - stdDay: Daily standard deviation of daytime values.
    - stdEvening: Daily standard deviation of evening values.
    - stdNight: Daily standard deviation of nighttime values.
    - skew: Daily skewness of the load profile.
    - quan25: Daily 25th percentile of the load profile.
    - quanDist: Daily difference between the 75th and 25th percentiles of the load profile.
    - loadFactor: Daily load factor of the load profile.
    - coincidenceFactor: Daily coincidence factor of the load profile.
    - peakValley: Daily ratio of the difference between the peak and valley values to the valley value.
    - pLoadFactor: Daily peak load factor of the load profile.
    - fLoadFactor: Daily flat load factor of the load profile.
    - vLoadFactor: Daily valley load factor of the load profile.
    - morning: Daily ratio of morning values to average.
    - evening: Daily ratio of evening values to average.
    - meanChange: Daily mean absolute change.
    - nonLinear: Daily non-linear metric.
    - linReg: Regression coefficient between neighboring days.
    - peakNo: Daily number of peaks in the load profile.
    - peakLoc: Hourly location of peaks in the load profile.
    - peakMag: Hourly magnitude of peaks in the load profile.
    - peakWidth: Hourly width of peaks in the load profile.

    Notes:
    - This function uses Min-Max scaling to normalize the features in the DataFrame.
    - It returns a DataFrame with the same structure as the input DataFrame, but with normalized feature values.
    """

    scaler = MinMaxScaler()
    features = ['mean', 'std', 'avMorning', 'avDay', 'avEvening', 'avNight', 'stdMorning', 'stdDay', 'stdEvening',
                'stdNight', 'skew', 'quan25', 'quanDist', 'loadFactor', 'coincidenceFactor', 'peakValley',
                'pLoadFactor', 'fLoadFactor', 'vLoadFactor', 'morning', 'evening', 'meanChange', 'nonLinear', 'linReg',
                'peakNo', 'peakLoc', 'peakMag', 'peakWidth']
    # deepcopy of dataframe
    df_copy = copy.deepcopy(df_param)

    df_n_feature = pd.DataFrame()
    for f in features:
        # normalization
        df_n_feature = pd.concat([df_n_feature, pd.DataFrame(
            scaler.fit_transform(np.array(df_copy.iloc[df_param.index.get_loc(f)])).transpose()).transpose()])

    df_n_feature.columns = df_copy.columns
    df_n_feature.index = df_copy.index

    return df_n_feature
