"""
Created on Mon Oct 31 10:49:49 2022

"""
import argparse
from typing import Optional
import pandas as pd
from sqlalchemy import create_engine, engine
import configparser
import os
from pathlib import Path

# requires psycopg2 2.9.5 - PostgreSQL Database Adapter to work


def get_ts_snh(sensor_id = None, 
               time_resolution= '15min',
               ts_from = '2017-01-01 00:00:00+01', 
               ts_to = '2022-07-21 23:45:00+02',
               timezone = 'Europe/Berlin',
               engine=False,
               tablename='smartmeter_snh' 
                        ):
    
    """
        Downloading SNH Smart Meter data
        from the time series database (applik-d208/last_secure) 

        Parameter
        ---------
        sensor_id (list) : id of SNH Smart Meter
        time_resolution (string): time resolution of the data
        ts_from (string): start time of the data
        ts_to (string): end time of the data
        engine :pointing to last_secure db on (applik-d208/last_secure)
        tablename (string) : 'smartmeter_snh' or 'smartmeter_data4grid_snh'

        Return
        ---------
        ts_values (pandas dataframe): time series data
    """

    
    



    # returns all devices when no SNH_addresse is supplied
    if sensor_id == None or len(sensor_id) == 0:
        sensor_query = ""
    else:
        sensor  = "'"+"','".join(sensor_id)+"'"
        sensor_query = f""" AND "id" IN ({sensor}) """


    # query_meta = (f"""SELECT id, timezone """
    #               f"""FROM public.smartmeter_snh_meta """
    #               f"""WHERE id>0 {sensor_query}"""
    #               )
    
    # meta_data = pd.read_sql_query(query_meta, engine)

    # 

    sql1 = (f""" SELECT id, count(id) as anzahl_id
                FROM {tablename}
                WHERE 1=1 {sensor_query} 
                GROUP BY id
                order by id""")
    
    id_info = pd.read_sql_query(sql1, engine)


    id_ok = id_info.loc[id_info['anzahl_id'] > 1, 'id'].astype(str).tolist()
    
    sensor  = "'"+"','".join(id_ok)+"'"
    sensor_query = f""" AND "id" IN ({sensor}) """
    

    sql2 = (f""" SELECT time_bucket_gapfill('{time_resolution}', "time", '{timezone}') as time, 
                        interpolate(avg("power_w")) AS W, id 
                FROM  {tablename}
                WHERE "time" BETWEEN '{ts_from}' AND '{ts_to}' {sensor_query} 
                GROUP BY time_bucket_gapfill('{time_resolution}', "time", '{timezone}'), id 
                ORDER BY id,time""")

    # print(sql2)
    ts_values = pd.read_sql_query(sql2, engine)

    return ts_values



def get_sensor_id(engine=False,tablename='snh_smartmeter', ts_from: Optional[str]=None, ts_to: Optional[str]=None):
    """
        Gets all sensor_id addresses of devices in SNH Data.
        When ts_from and ts_to are supplied only retrieves from this range.
        Args:
            engine: the SQL connection to use (from SNH class)
            ts_from: start of the time series (str in date time format)
            ts_to: end of the time series (str in date time format)
            tablename (string) : 'smartmeter_snh' or 'smartmeter_data4grid_snh'
        Returns:
            list of sensor ids in the SNH dataset
    """
    if ts_from != None and ts_to != None:
        where_query = f"WHERE time BETWEEN '{ts_from}' AND '{ts_to}'"
    else:
        where_query = ''

    query = (
        f'SELECT id',
        f'FROM {tablename}',
        where_query
    )

    query = "\n".join(query)
    values = pd.read_sql_query(query, engine)

    return values

def get_meta_snh(engine=False,tablename='snh_smartmeter_meta'):
    """
        Gets all id  of devices in snh Data with meta data as PLZ and id of weathermodel
        
        Args:
            engine: the SQL connection to use (last_secure)
            tablename (string) : 'snh_smartmeter_meta' or 'smartmeter_data4grid_snh_meta'
        Returns:
            list of ids, plz and weathermodel_id in the snh dataset with meta data
    """

    query = (
        f'SELECT id, plz, id_era5_land ',
        f'FROM public.{tablename} '
    )

    query = "\n".join(query)

    values = pd.read_sql_query(query, engine)

    return values



def get_weather_data(id_weathermodel=[],
                    time_resolution= '1h',
                    ts_from = '2017-01-01 00:00:00+01', 
                    ts_to = '2022-07-21 23:45:00+02',
                    timezone = 'Europe/Berlin',
                    engine = False
                    ):
    """
        Returns weatherdata of given weathermodel id in a given time period

        Args:
            ts_from: start of the time series (str in date time format)
            ts_to: end of the time series (str in date time format)
            id_weathermodel: list with indexes of weathermodel pixel
            engine :pointing to last_secure db on (applik-d208/last)
        Returns:
            dataframe with weatherdata: time = timestamp
                                        id_era5_land = id of weather model raster
                                        t2m = temperature 2 m above ground in K
                                        ghi = global horizontal irradiation in W/m²
                                        ws10 = windspeed 10 m above ground in m/s
    """

    # returns all devices when no SNH_addresse is supplied
    if id_weathermodel == None or len(id_weathermodel) == 0:
        id_query = ""
    else:
        # Alle Werte in id_list in Strings umwandeln
        str_id_list = [str(value) for value in id_weathermodel]
        id  = "'"+"','".join(str_id_list)+"'"
        id_query = f""" AND "id" IN ({id}) """



    sql = (f""" SELECT time AT TIME ZONE '{timezone}', id as id_era5_land, t2m,  ghi, ws10  
                FROM (  
                    SELECT time_bucket_gapfill('{time_resolution}', "time") as time, 
                    interpolate(avg(t2m)) as t2m, interpolate(avg(ghi)) as ghi, interpolate(avg(ws10)) as ws10, id 
                    FROM  public.era5_land 
                    WHERE "time" BETWEEN '{ts_from}' AND '{ts_to}' {id_query} 
                    GROUP BY time_bucket_gapfill('{time_resolution}', "time"), id 
                    ) AS subquery 
                ORDER BY id,time""")

    ts_weather = pd.read_sql_query(sql, engine)

    return ts_weather



if __name__ == '__main__':
    
    #TODO: add last_secure to last_db_ini
    
    userAG = str(Path.home()) == 'C:\\Users\\agoldmaier' # True if user is Ann-Katrin Goldmaier

    parser = argparse.ArgumentParser()
    config = configparser.ConfigParser()
    
    if userAG:
        config.read(os.path.join(Path.home(), "iee.ini"))    
    else:
        parser.add_argument('--db-ini', default='last_db.ini', help='path to db .ini file.')
        args = parser.parse_args()
        config.read(Path(args.db_ini))
    
    pgsql_secure = dict(config["last_secure"])
    pgsql_last = dict(config["last"])

    if userAG:
        pgsql_secure['user'] = pgsql_secure['username']
        pgsql_last['user'] = pgsql_last['username']


    try:
        engine_secure = create_engine(f"postgresql://{pgsql_secure['user']}:"
                                f"{pgsql_secure['password']}"
                                f"@{pgsql_secure['host']}/{pgsql_secure['database']}")    
    except Exception as e:
        print(f'Verbindung zur DB nicht möglich:{e}')
        quit()  
    

    all_ids = ['{0}'.format(i) for i in range(1,177)]
    ids = ['{0}'.format(i) for i in range(24,27)]

    hamburg_smart = {
                    'sensor_id': ['24'],#['24','25','26'],
                    'time_resolution': '15min',
                    'ts_from': '2021-01-01 01:45:00+01' ,
                    'ts_to': '2021-01-03 01:45:00+01',
                    'timezone': 'UTC',
                    'tablename':'smartmeter_data4grid_snh'
                        }
    
    hamburg_daten = get_ts_snh(
                               sensor_id = hamburg_smart['sensor_id'], 
                               time_resolution= hamburg_smart['time_resolution'],
                            #    ts_from = hamburg_smart['ts_from'],
                            #    ts_to = hamburg_smart['ts_to'],
                               timezone = hamburg_smart['timezone'],
                               engine= engine_secure,
                               tablename=hamburg_smart['tablename']
                               )

    print(hamburg_daten.to_string())
    print(hamburg_daten.head())

    meta_data = get_meta_snh(engine= engine_secure,tablename=hamburg_smart['tablename']+'_meta')
    print(meta_data.head())


    engine_secure.dispose()

    # ## requests of weatherdata from last db requires other engine
    try:
        engine_last = create_engine(f"postgresql://{pgsql_last['user']}:"
                                f"{pgsql_last['password']}"
                                f"@{pgsql_last['host']}/{pgsql_last['database']}")    
    except Exception as e:
        print(f'Verbindung zur DB nicht möglich:{e}')
        quit()  

    # id of weathermodel
    # id_wm = list(meta_data['id_era5_land'].drop_duplicates())

    

    id_wm= list(meta_data.loc[(meta_data['id'].isin([24,50]) ),'id_era5_land'].values)


    weatherdata = get_weather_data(id_weathermodel=id_wm,
                    time_resolution= '1h',
                    ts_from = '2020-12-31 22:00:00+01', 
                    ts_to = '2022-07-21 23:45:00+02',
                    timezone = 'UTC',
                    engine = engine_last
                    )

    print(weatherdata.head())

    engine_last.dispose()