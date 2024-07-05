"""
Created on Mon Oct 31 10:49:49 2022

"""
import argparse
from typing import Optional
import pandas as pd
from sqlalchemy import create_engine, engine
import configparser
from pathlib import Path
import os
import numpy as np

# requires psycopg2 2.9.5 - PostgreSQL Database Adapter to work


def get_ts_wpuq(sensor_id = ['sfh3'], 
               time_resolution= '10sec',
               ts_from = '2018-01-01 00:00:00+00', 
               ts_to = '2020-12-31 23:59:50+00',
               timezone = 'Europe/Berlin',
               engine=False
                        ):
    """
        Downloading wpuq Smart Meter data
        from the time series database (applik-d208/last) 

        Parameter
        ---------
        sensor_id (list) : id of household sfh3 - sfh40
        time_resolution (string): time resolution of the data
        ts_from (string): start time of the data
        ts_to (string): end time of the data
        engine :pointing to last_secure db on (applik-d208/last_secure)

        Return
        ---------
        ts_values (pandas datafram): time series data
    """

    

    # returns all devices when no wpuq_addresse is supplied
    if sensor_id == None or len(sensor_id) == 0:
        sensor_query = ""
    else:
        sensor  = "'"+"','".join(sensor_id)+"'"
        sensor_query = f""" AND "id" IN ({sensor}) """



    # 
    sql = (f""" SELECT time at time zone '{timezone}' as time, W, id
                FROM(
                    SELECT time_bucket_gapfill('{time_resolution}', "time") as time, 
                            interpolate(avg("p_tot")) AS W, id 
                    FROM  public.wpuq_ts
                    WHERE "time" BETWEEN '{ts_from}' AND '{ts_to}' {sensor_query} 
                    GROUP BY time_bucket_gapfill('{time_resolution}', "time"), id 
                    ) as subquery
                ORDER BY id,time""")

    ts_values = pd.read_sql_query(sql, engine)

    return ts_values

def get_meta_wpuq(engine: engine.Engine):
    """
        Gets all sensor_id addresses of devices in wpuq Data with meta data as inhabitants, living space in m² and if PV is installed.
        
        Args:
            engine: the SQL connection to use (from wpuq class)
        Returns:
            list of household ids in the wpuq dataset with meta data
    """

    query = (
        f'SELECT id, n_inhabitants, living_space_m2, pv, "pv_system_size_kWp", ventilation_syst, "PLZ" ,id_era5_land',
        f'FROM wpuq_meta'
    )

    query = "\n".join(query)
    values = pd.read_sql_query(query, engine, dtype={"n_inhabitants": np.float32, "living_space_m2": np.float32})#, "pv_system_size_kWp": np.float32})

    return values

if __name__ == '__main__':

    userAG = str(Path.home()) == 'C:\\Users\\agoldmaier' # True if user is Ann-Katrin Goldmaier

    parser = argparse.ArgumentParser()
    config = configparser.ConfigParser()
    
    if userAG:
        config.read(os.path.join(Path.home(), "iee.ini"))
    else:
        parser.add_argument('--db-ini', default='last_db.ini', help='path to db .ini file.')
        args = parser.parse_args()
        config.read(Path(args.db_ini))
    
    pgsql_info = dict(config["last"])

    if userAG:
        pgsql_info['user'] = pgsql_info['username']
    
    

    try:
        engine = create_engine(f"postgresql://{pgsql_info['user']}:"
                                f"{pgsql_info['password']}"
                                f"@{pgsql_info['host']}/{pgsql_info['database']}")    
    except Exception as e:
        print(f'Verbindung zur DB nicht möglich:{e}')
        quit()  
    
    wpuq_input = {
                    'sensor_id': ['sfh4','sfh10','sfh33'],
                    'time_resolution': '5sec',
                    'timezone': 'UTC',
                    'ts_from': '2019-03-31 01:45:00+01' ,
                    'ts_to': '2019-04-01 01:45:00+01'
                        }
    
    wpuq_daten = get_ts_wpuq(
                               sensor_id = wpuq_input['sensor_id'], 
                               time_resolution= wpuq_input['time_resolution'],
                               ts_from = wpuq_input['ts_from'],
                               ts_to = wpuq_input['ts_to'],
                               timezone = wpuq_input['timezone'],
                               engine= engine
                               )

    print(wpuq_daten.to_string())

    metadata = get_meta_wpuq(engine)

    print(metadata.to_string())

