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

def get_ts_loandonsmartmeter(sensor_id = ['MAC005237'], 
                            time_resolution= '30min',
                            ts_from = '2011-11-30T01:00:00Z', 
                            ts_to = '2014-11-30T24:00:00Z',
                            engine=False
                        ):
    """
        Downloading London Smart Meter data
        from the time series database (applik-d2008/last) 

        Parameter
        ---------
        mca_adresse (list) : id of London Smart Meter
        time_resolution (string): time resolution of the data
        ts_from (string): start time of the data
        ts_to (string):
        engine :pointing to last db on (applik-d2008/last)

        Return
        ---------
        ts_values (pandas datafram): time series data
    """

    # returns all devices when no mca_addresse is supplied
    if sensor_id == None or len(sensor_id) == 0:
        mca_query = ""
    else:
        mcas = "'"+"','".join(sensor_id)+"'"
        mca_query = f""" AND "LCLid" IN ({mcas}) """


    sql = (f""" SELECT time_bucket_gapfill('{time_resolution}', "DateTime") """
            f""" as time, avg("kwh/hh") as KWh_hh, "LCLid" as id FROM """
            f""" cc_clc_fulldate_raw  WHERE """
            f""" "DateTime" BETWEEN '{ts_from}' AND '{ts_to}' """
            f"""{mca_query}"""
            f""" GROUP BY time_bucket_gapfill('{time_resolution}', "DateTime"), """
            f""" "LCLid" ORDER BY time""")

    ts_values = pd.read_sql_query(sql, engine)

    return ts_values

def get_macs(engine: engine.Engine, ts_from: Optional[str]=None, ts_to: Optional[str]=None):
    """
        Gets all MAC addresses of devices in LSM.
        When ts_from and ts_to are supplied only retrieves from this range.
        Args:
            engine: the SQL connection to use (from LondonSmartMeter class)
            ts_from: start of the time series (str in date time format)
            ts_to: end of the time series (str in date time format)
        Returns:
            list of LCLid in the LSM dataset
    """
    if ts_from != None and ts_to != None:
        where_query = f'WHERE "DateTime" BETWEEN "{ts_from}" AND "{ts_to}"'
    else:
        where_query = ''

    query = (
        f'SELECT "LCLid"',
        f'FROM cc_clc_folldate_raw',
        where_query
    )

    query = "\n".join(query)
    values = pd.read_sql_query(query, engine)

    return values

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--db-ini', default='last_db.ini', help='path to db .ini file.')
    args = parser.parse_args()


    config = configparser.ConfigParser()
    config.read(Path(args.db_ini))
    pgsql_info = dict(config["last"])

    try:

        engine = create_engine(f"postgresql://{pgsql_info['user']}:"
                                f"{pgsql_info['password']}"
                                f"@{pgsql_info['host']}/{pgsql_info['database']}")    
    except Exception as e:
        print(f'Verbingung zur DB nicht möglich:{e}')
        quit()  

    london_smart = {
                    'mca_adresse': ['MAC005237',
                                    'MAC004422',
                                    'MAC002417'
                                    ],
                        'time_resolution': '30min',
                        'ts_from': '2011-11-30T01:00:00Z' ,
                        'ts_to': '2014-11-30T24:00:00Z'
                        }
    loadon_daten = get_ts_loandonsmartmeter(
                                            mca_addresse = london_smart['mca_adresse'], 
                                            time_resolution= london_smart['time_resolution'],
                                            ts_from = london_smart['ts_from'],
                                            ts_to = london_smart['ts_to'],
                                            engine= engine
                                            )

    print(loadon_daten.to_string())

