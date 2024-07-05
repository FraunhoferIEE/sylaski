import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def min_max_scale_data(df: pd.DataFrame, ivl=(0, 1)):
    """
    Scales each feature to the given range using min max scaling.
    Args:
        df: dataframe with data of individual households as columns
        ivl: scaling interval. Default is [0,1].
    Returns:
        df_scaled: scaled dataframe
    """
    scaler = MinMaxScaler(feature_range=ivl)
    df_transformed = scaler.fit_transform(df)

    return df_transformed


if __name__ == '__main__':
    df = pd.DataFrame([[0,2], [0,3]])
    df[df.columns] = min_max_scale_data(df[df.columns])
    print(df.columns)
