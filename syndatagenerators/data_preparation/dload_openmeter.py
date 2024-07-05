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


def get_ts_openmeter(sensor_id = ['9415ae40-f716-4e78-88b9-51b17fe8acc9'], 
               time_resolution= '15min',
               ts_from = '', 
               ts_to = '',
               timezone = 'Europe/Berlin',
               engine=False
                        ):
    """
        Downloading openmeter data
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


    # returns all devices when no openmeter_addresse is supplied
    if sensor_id == None or len(sensor_id) == 0:
        sensor_query = ""
    else:
        sensor  = "'"+"','".join(sensor_id)+"'"
        sensor_query = f""" AND "sensor_id" IN ({sensor}) """


    # returns whole timeseries when no start or end time is supplied
    if ts_from == None or len(ts_from) == 0 or ts_to == None or len(ts_to) == 0:
       
        query = f"""SELECT MIN("start_time") AS min_time, MAX("start_time") AS max_time
                        FROM public.openmeter_measurement_ts
                        WHERE 1=1 {sensor_query}"""
        
        min_max_time = pd.read_sql_query(query, engine)
        ts_from = str(min_max_time.loc[0,'min_time'])
        ts_to = str(min_max_time.loc[0,'max_time'])
        # print(ts_from)
        # print(ts_to)
       
    time_query = f""" "start_time" BETWEEN '{ts_from}' AND '{ts_to}'  """

    sql = f""" SELECT time at time zone '{timezone}' as time, W, sensor_id as id
                FROM ( 
                    SELECT time_bucket_gapfill('{time_resolution}', "start_time") as time, 
                            interpolate(avg("average_power_w")) AS W, sensor_id 
                    FROM  public.openmeter_measurement_ts 
                    WHERE  {time_query} {sensor_query}
                    GROUP BY time_bucket_gapfill('{time_resolution}', "start_time"), sensor_id 
                    ) AS subquery
                ORDER BY sensor_id,time"""


    ts_values = pd.read_sql_query(sql, engine)

    return ts_values

def get_meta_openmeter(engine: engine.Engine,meta4ts=False):
    """
        Gets all sensor_id addresses of devices in openmeter Data with meta data as inhabitants, living space in m² and if PV is installed.
        
        Args:
            engine: the SQL connection to use (from openmeter class)
            meta4ts: bool true, for getting only metadata for sensors where timeseries are in db
        Returns:
            list of sensor_ids in the openmeter dataset with meta data
    """
    federal_state = ['Bremen','Hamburg','Mecklenburg-Vorpommern','Niedersachsen',
                     'Rheinland-Pfalz','Saarland','Sachsen','Sachsen-Anhalt','Berlin','Nordrhein-Westfalen',
                     'Baden-Württemberg','Bayern','Brandenburg','Hessen','Schleswig-Holstein','Thüringen']
    fs  = "'"+"','".join(federal_state)+"'"

    if meta4ts:
        query = (
        f'''SELECT sensor_id,measurement_category, measures_from, measures_to, notes, area, category, usage, usage_detail, construction_year,
                    erzeuger_vorhanden, city, federal_state, country, post_code, location_id, lat_plz, lon_plz, geom_plz, coords_matching_bl, 
                    id_era5_land, ts_vorhanden
            FROM public.openmeter_meta_mit_erz 
            WHERE federal_state IN ({fs}) AND measures_from is not NULL AND ts_vorhanden;'''
        )
    else:
        query = (
            f'''SELECT sensor_id,measurement_category, measures_from, measures_to, notes, area, category, usage, usage_detail, construction_year,
                        erzeuger_vorhanden, city, federal_state, country, post_code, location_id, lat_plz, lon_plz, geom_plz, coords_matching_bl, 
                        id_era5_land, ts_vorhanden
                FROM public.openmeter_meta_mit_erz 
                WHERE federal_state IN ({fs}) AND measures_from is not NULL;'''
        )
    # print(query)

    values = pd.read_sql_query(query, engine)

    print(f'''
          INFOS ZU METADATEN VON OPENMETER:
          measurement_category: Erzeugung oder Verbrauch 
          erzeuger_vorhanden: true bei Verbraucher, wenn es mindestens einen Erzeuger mit der selben location_id gibt 
          lat_plz,lon_plz,geom_plz: zugeordnet, nach Schwerpunkten der PLZ-Gebiete 
          coords_matching_bl: true, wenn Koordinaten zu Bundesland passen, sonst wahrscheinlich falsche Wetterdaten''')

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
    
    openmeter_input = {
                    'sensor_id': ['9415ae40-f716-4e78-88b9-51b17fe8acc9'],#[],#
                    'time_resolution': '15min',
                    'timezone': 'UTC',
                    'ts_from': '2020-09-23' ,
                    'ts_to': '2020-11-23'
                        }
    
    
    meta_data = get_meta_openmeter(engine,meta4ts=True)
    
    print(meta_data.to_string())
    
    meta_data = get_meta_openmeter(engine)

    print(meta_data.to_string())
    
    openmeter_daten = get_ts_openmeter(
                               sensor_id = openmeter_input['sensor_id'], 
                               time_resolution= openmeter_input['time_resolution'],
                               ts_from = openmeter_input['ts_from'],
                               ts_to = openmeter_input['ts_to'],
                               timezone = openmeter_input['timezone'],
                               engine= engine
                               )

    print(openmeter_daten.to_string())



    

    engine.dispose()