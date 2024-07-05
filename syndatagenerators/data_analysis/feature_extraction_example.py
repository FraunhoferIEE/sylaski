# extraction of statistical features from load profiles

# import modules
import pandas as pd
import syndatagenerators.data_analysis.feature_extraction as feature_extraction


## path to data (.csv)
load_profile_data = r"C:\Users\lriedl\Documents\1_Projekte\SyLas-KI\Daten\OM_subset_Final.csv"

# load and prepare data
load_profile_data = pd.read_csv(load_profile_data)
load_profile_data.index = pd.to_datetime(load_profile_data.pop('time'))

## create average weeks
df_averaged_load_data = feature_extraction.average_weeks(load_profile_data)

## calculate statistical features of averaged weeks
statistical_features = feature_extraction.get_param(df_averaged_load_data)
