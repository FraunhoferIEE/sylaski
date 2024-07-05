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

# requires psycopg2 2.9.5 - PostgreSQL Database Adapter to work


def get_ts(sensor_id, 
               time_resolution= '15min',
               ts_from = '2021-01-01 00:00:00+01', 
               ts_to = '2022-01-01 00:00:00+01',
               engine=False
                        ):
    """
        Downloading chargingstation data
        from the time series database (applik-d208/last) 

        Parameter
        ---------
        sensor_id (list) : id of sensor, get all available sensors, when input is empty
        time_resolution (string): time resolution of the data
        ts_from (string): start time of the data (empty string if all data shall be loaded)
        ts_to (string): end time of the data (empty string if all data shall be loaded)
        engine :pointing to last_secure db on (applik-d208/last_secure)

        Return
        ---------
        ts_values (pandas datafram): time series data
    """


    # returns all devices when no sensor_id is supplied
    if sensor_id == None or len(sensor_id) == 0:
        sensor_query = ""
    else:
        sensor  = "'"+"','".join(sensor_id)+"'"
        sensor_query = f""" AND "sensor_id" IN ({sensor}) """


    # returns whole timeseries when no start or end time is supplied
    if ts_from == None or len(ts_from) == 0 or ts_to == None or len(ts_to) == 0:
       
        query = f"""SELECT MIN("start_time") AS min_time, MAX("start_time") AS max_time
                        FROM public.chargingstation
                        WHERE 1=1 {sensor_query}"""
        
        min_max_time = pd.read_sql_query(query, engine)
        ts_from = str(min_max_time.loc[0,'min_time'])
        ts_to = str(min_max_time.loc[0,'max_time'])
       
    time_query = f""" "start_time" BETWEEN '{ts_from}' AND '{ts_to}'  """

    sql = f""" SELECT time at time zone 'Europe/Berlin' as time, W, sensor_id as id
                FROM ( 
                    SELECT time_bucket_gapfill('{time_resolution}', "start_time") as time, 
                            interpolate(avg("average_power_w")) AS W, sensor_id 
                    FROM  public.chargingstation 
                    WHERE  {time_query} {sensor_query}
                    GROUP BY time_bucket_gapfill('{time_resolution}', "start_time"), sensor_id 
                    ) AS subquery
                ORDER BY sensor_id,time"""


    ts_values = pd.read_sql_query(sql, engine)

    return ts_values