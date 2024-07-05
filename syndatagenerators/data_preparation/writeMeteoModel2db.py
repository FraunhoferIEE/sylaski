# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 11:27:35 2022

@author: agerlach
"""
#%% import
import netCDF4 as nc
from typing import Optional
import numpy as np
import pandas as pd
import configparser
from sqlalchemy import create_engine, text
import psycopg2
from pathlib import Path
import os
import sys
import requests

# import energyants
sys.path.append(r'C:\Users\agoldmaier\Documents\GitLabFhG\energyants')
# sys.path.append(r'/share/data1/agerlach/GitLabFhG/energyants')
from energyants.meteo.models.era5land.model import Era5Land
from energyants.meteo.models.sarah3.model import Sarah3




def get_engine(database):
    
    config = configparser.ConfigParser()
    config.read(os.path.join(Path.home(), "iee.ini"))
    pgsql = dict(config[database])

    db_url_data= (f"postgresql://{pgsql['username']}:{pgsql['password']}"
            f"@{pgsql['host']}/{pgsql['database']}")

    # Verbindungen zur PostgreSQL-Datenbank herstellen
    return create_engine(db_url_data)


def write_centroids_in_db(filepath,tablename,engine):
    """
    writes centroids of a weathermodel to the last Database

    Parameters
    ----------
    filepath : str - filepath to nc data-file that contains latitude and longitude of weather model
    tablename : str - name of table in database
     : 

    Returns
    -------
    
    """
     
    # Pixel aus Wettermodell extrahieren

    with nc.Dataset(filepath) as ds:
        lons = np.array(ds["longitude"])
        lats = np.array(ds["latitude"])

    # 1. Lookup-Table für Koordinaten: Kombination von allen lat mit allen lon Werten

    lat_lon = [(lat, lon) for lat in lats for lon in lons]

    weathermodel = pd.DataFrame(lat_lon)
    
    weathermodel.rename(columns={0: "lat",1: "lon"}, inplace=True)
    print(weathermodel.columns)
    print(weathermodel.head())
    # Lookup-Table in die Datenbank speichern

    weathermodel.to_sql(name=tablename, con=engine, schema='public', if_exists='replace')

    add_geom_column(tablename,engine)
    


def add_geom_column(tablename,engine,name_geom='geom',name_lat='lat',name_lon='lon'):
     
     # Geometriespalte hinzufügen:
    query = f'''ALTER TABLE {tablename} 
                ADD COLUMN {name_geom} geography(Point,4326);
                
                UPDATE {tablename} 
                SET {name_geom} = ST_SetSRID(ST_MakePoint({name_lon}, {name_lat}), 4326);'''
    
    with engine.connect() as connection:
            connection.execute(text(query))
            connection.commit()  # commit the transaction

    return name_geom


     



def find_entrys_wo_weatherdata(dataset_ts,dataset_meta,dataset_id,dataset_time,name_of_index_wm,engine):
    """
    Selects entrys of timeseries Datatable, that don't have a connected timeseries of the chosen
    weathermodel.

    Parameters
    ----------
    dataset : str - name of database table with timeseries
    dataset_id : str - name id column, which connects timeseries with meta data
    name_of_index_wm: str - name of column that contains the index of the chosen weathermodel

    Returns
    -------
    entrys_wo_weather: dataframe - all timestamps of timeseries, without weatherdata
    """

    
    # Prüfen, für welche Lastzeitreihen noch keine Wetterdatenzuordnung vorhanden ist.

    query = f'''
            SELECT 
                data.{dataset_id} as id, min(data.{dataset_time} at time zone 'UTC') as mintime,
                  max(data.{dataset_time} at time zone 'UTC') as maxtime, meta.{name_of_index_wm} as index_wm
            FROM public.{dataset_ts} as data, public.{dataset_meta} as meta
            WHERE data.{dataset_id} = meta.{dataset_id}
            GROUP BY data.{dataset_id}, index_wm;
            '''
    
    print(query)


    entrys_wo_weather = pd.read_sql_query(query,engine)


    return entrys_wo_weather


#%%  Identifizieren der unterschiedlichen Standorte des Datensatzes
def find_places_from_meta(dataset,dataset_id,list_of_entrys,engine):
    # standorte aus wpuq meta abfragen von daten, die keine Wetterdaten haben.

    
    ids  = "'"+"','".join(list_of_entrys)+"'"
    query = (
        f"SELECT {dataset_id},geom FROM {dataset} "
        f"WHERE {dataset_id} IN ({ids}) "
    )

    places = pd.read_sql_query(query,engine)
    
    # print(places)

    # unique_places = places[['lat','lon']].drop_duplicates()

    return places




## Zuordnung Standorte zu Wetterpixeln (sql?)
# die Zuordnung des Wetterpixelindexes in der Tabelle der Lastinfos ergänzen

def add_index_weatherdata(dataset,dataset_id,dataset_wm_index,table_wm,name_of_index_wm,engine,column_geom='geom'):
    """
    Find which weather data pixel has shortest distance to locations in meta data and
    add index in meta data table

    Parameters
    ----------
    dataset : str - name of database table with meta data
    dataset_id : str - name id column of dataset
    dataset_wm_index : str - name of column that contains the index of smallest distance from the weathermodel table 
    table_wm : str - name of weather model table
    name_of_index_wm: str - name of column that contains the index of the chosen weathermodel
    engine:  - database connection

    Returns
    -------
    places_with_wm_index: dataframe - input df with weathermodel index
    """

    query = (
        f"SELECT {dataset_id} as id "
        f"FROM {dataset} "
        f"WHERE {dataset_wm_index} is NULL"
    )

    id_pd = pd.read_sql_query(query,engine)
    print(type(id_pd))
    print(id_pd.head())
    id_list = list(id_pd['id'])
    # Alle Werte in id_list in Strings umwandeln
    str_id_list = [str(value) for value in id_list]
    # IDs mit Komma getrennt hintereinander schreiben
    ids  = "'"+"','".join(str_id_list)+"'"

## TODO: STATEMENT TESTEN. EVTL VORHER INDEX ERSTELLEN SIEHE SQL-SKRIPT
#        ("KLEINSTE DISTANZ ULTIMATIVE LÖSUNG")
# #%% Zuordnung der Lastzeitreihen standorte zu index aus Wettermodell
    query = (
         f"""
         UPDATE {dataset} 
         SET {dataset_wm_index} = closest_table2_id
         FROM
            (SELECT  {dataset}.{dataset_id} AS table1_id,
                    ( SELECT {table_wm}.{name_of_index_wm}
                        FROM {table_wm}
                        ORDER BY {dataset}.{column_geom} <-> {table_wm}.geom
                        LIMIT 1
                    ) AS closest_table2_id
            FROM (SELECT * 
                  FROM {dataset} 
                  WHERE {dataset_wm_index} IS NULL
                  ) as {dataset}
            ) as dist
         WHERE {dataset}.{dataset_id} = dist.table1_id"""
    )
    
    # query = (
    #     f"UPDATE {dataset} "
    #     f"SET {dataset_wm_index} = dist.nearest_wm_id "
    #     f"FROM (SELECT "
    #     f"        DISTINCT ON (dataset.{dataset_id}) dataset.{dataset_id} as id, weathermodel.{name_of_index_wm} as nearest_wm_id, "
    #     f"        ST_Distance(dataset.geom, weathermodel.geom) AS distance "
    #     f"    FROM {dataset} as dataset, {table_wm} as weathermodel "
    #     f"    WHERE dataset.{dataset_id} IN ({ids}) "
    #     f"    ORDER BY dataset.{dataset_id}, ST_Distance(dataset.geom, weathermodel.geom)) AS dist "
    #     f"WHERE {dataset}.{dataset_id} = dist.id"
    # )

    print(query)

    with engine.connect() as connection:
            connection.execute(text(query))
            connection.commit()  # commit the transaction
    

def create_table_weatherdata(tablename_weatherdata,engine):

    # config = configparser.ConfigParser()
    # config.read(os.path.join(Path.home(), "iee.ini"))
    # pgsql_info = dict(config['last'])

    # # Verbindung zur PostgreSQL-Datenbank herstellen
    # conn = psycopg2.connect(
    #     dbname="last",
    #     user=pgsql_info['userame'],
    #     password=pgsql_info['password'],
    #     host=pgsql_info['host'],
    #     port=pgsql_info['port']
    # )

    # # Cursor erstellen
    # cur = conn.cursor()

    query1 = (
        f"CREATE TABLE {tablename_weatherdata} ("
        f"time TIMESTAMPTZ NOT NULL,"
        f"id INT NOT NULL,"
        f"t2m DOUBLE PRECISION," # [K]
        f"ghi DOUBLE PRECISION," # [W/m²]
        f"ws10 DOUBLE PRECISION," # [m/s]
        #-- Füge ggf. weitere Spalten hinzu "
        f"PRIMARY KEY (time, id)"
        f");"
    )

    with engine.connect() as connection:
            connection.execute(text(query1))
            connection.commit()  # commit the transaction

    query2 = (
        f"SELECT create_hypertable('{tablename_weatherdata}', 'time');"
        )

    with engine.connect() as connection:
            connection.execute(text(query2))
            connection.commit()  # commit the transaction
    

    print('Ende der Funktion')




def find_missing_years_and_fill(dataframe,engine):
    # Verbindung zur PostgreSQL-Datenbank herstellen
    # config = configparser.ConfigParser()
    # config.read(os.path.join(Path.home(), "iee.ini"))
    # pgsql_info = dict(config['last'])

    # # Verbindung zur PostgreSQL-Datenbank herstellen
    # conn = psycopg2.connect(
    #     dbname="last",
    #     user=pgsql_info['username'],
    #     password=pgsql_info['password'],
    #     host=pgsql_info['host'],
    #     port=pgsql_info['port']
    # )

    # # Cursor erstellen
    # cursor = conn.cursor()

    # Leerer DataFrame zum Speichern der fehlenden Jahre pro Index
    missing_years_df = pd.DataFrame(columns=['index_wm', 'Missing Years'])

    # Iteriere über jede Zeile im DataFrame
    for index, row in dataframe.iterrows():
        id = row['id']
        index_wm = row['index_wm']
        min_time = row['mintime']
        max_time = row['maxtime']
        
        # print(id)
        # SQL-Abfrage, um die fehlenden Jahre für den aktuellen Index zu finden
        query = f"SELECT DISTINCT EXTRACT(year FROM time) AS year FROM era5_land WHERE id = '{index_wm}' AND time >= '{min_time}' AND time <= '{max_time}'"
        
        result = pd.read_sql_query(query,engine)
        
        # cursor.execute(query)
        # result = cursor.fetchall()
        # Liste der vorhandenen Jahre für den aktuellen Index
        existing_years = list(result['year'])
        
        # existing_years = [int(year[0]) for year in result]

        # Liste der fehlenden Jahre für den aktuellen Index
        all_years = list(range(int(min_time.year), int(max_time.year)+1))
        
        missing_years = [year for year in all_years if year not in existing_years]      
        
        missing_years_df_temp = pd.DataFrame({'index_wm': index_wm, 'Missing Years': missing_years})
        
        if len(missing_years)>=1:
            print(query)
            print('result: ')
            print(result)
            print('existing_years: ')
            print(existing_years)
            print('all_years: ')
            print(all_years)        
            print('missing_years: ')
            print(missing_years)
            print('missing_years_df_temp: ')
            print(missing_years_df_temp)

        # Zeile zum DataFrame hinzufügen
        missing_years_df = pd.concat([missing_years_df,missing_years_df_temp])
        
    ## Datenbankverbindung schließen
    # cursor.close()
    # conn.close()

    # print(missing_years_df)

    return missing_years_df



def wetterapi(latitude,longitude,year,parameter,model,height=None):

    url = "http://applik-156.iee.fraunhofer.de:5555"
    
    header = {"x-access-token": "TOKEN"}
    
    region = "DEU"
    params = {"latitude": latitude, "longitude": longitude, "year": year,
                "region": region, "height": height,
                "parameters": parameter}
    
    era = (requests
        .get(url=f"{url}/meteo/{model}", params=params, headers=header)
        .json())
    print(era.keys())
    
    df = pd.DataFrame({k: v["data"] for k, v in era.items()})
    df.columns = [f"{k} [{v['unit']}]" for k, v in era.items()]

    return df
    


## für die fehlenden Pixel Wetterdaten ü#ber Wetterdatenloader laden
def get_weatherdata_from_model(df):
    
    
    years = df['Missing Years'].unique()
    
    df_weatherdata = pd.DataFrame(columns=['t2m','ghi','ws10','id'])

    # lat_list = [50.57]
    # lon_list = [12.53]
    # index_wm_list = [12345]

    for year in years:

        # wm = Era5Land(year=year, storage=r"/share/data3/meteo/era5land")
        wm = Era5Land(year=year, storage=r"C:\Users\agoldmaier\Documents\meteoData\test")
        # irr = Sarah3(year=year, storage=r"/share/data3/energyants-api/meteo/Sarah3")
        irr = Sarah3(year=year, storage=r"C:\Users\agoldmaier\Documents\meteoData\test")

        sub = df[df['Missing Years']== year]
        
        for i,row in sub.iterrows():

            id_wm = row['index_wm']
            lat = row['lat']
            lon = row['lon']


            t2m = wm.load(parameter="t2m", latitude=lat, longitude=lon)
            
            # ghi = wm.load(parameter="ghi", latitude=lat, longitude=lon)
            ghi = irr.load(parameter="ghi", latitude=lat, longitude=lon)

            ws10 = wm.load(parameter="ws10", latitude=lat, longitude=lon)
            
            df_ghi = pd.DataFrame({'ghi':ghi}, index=irr.dates)
            # print('1)')
            # print(df_ghi[df_ghi['ghi']<0])
            df_ghi.loc[df_ghi["ghi"] < 0, "ghi"] = np.nan
            df_ghi.interpolate(inplace=True)
            # print('2)')
            # print(df_ghi[df_ghi['ghi']<0])
            
            # df_temp = pd.DataFrame({'t2m':t2m, 'ghi':ghi, 'ws10':ws10},index = wm.dates)
            df_temp = pd.DataFrame({'t2m':t2m,'ghi':df_ghi['ghi'][irr.dates.isin(wm.dates)],  'ws10':ws10},index = wm.dates)
            
            df_temp['id'] = id_wm


            df_weatherdata = pd.concat([df_weatherdata,df_temp])

## TODO
#   /share/data1/agerlach/GitLabFhG/syndatagenerators/syndatagenerators/data_preparation/writeMeteoModel2db.py:428: 
#   FutureWarning: The behavior of array concatenation with empty entries is deprecated. 
#   In a future version, this will no longer exclude empty items when determining the result dtype. 
#   To retain the old behavior, exclude the empty entries before the concat operation.
#   df_weatherdata = pd.concat([df_weatherdata,df_temp])
#   /share/data1/agerlach/GitLabFhG/syndatagenerators/syndatagenerators/data_preparation/writeMeteoModel2db.py:428: 
#   FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. 
#   In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. 
#   To retain the old behavior, exclude the relevant entries before the concat operation.
#   df_weatherdata = pd.concat([df_weatherdata,df_temp])



    df_weatherdata = df_weatherdata.tz_localize('UTC', ambiguous='infer')

    return df_weatherdata



def get_coords_weathermodel(index_wm_list,table_weathermodel,name_of_index_wm,engine):
    
    
    ix_wm  = ",".join([str(_) for _ in index_wm_list])
    query = f"SELECT {name_of_index_wm} as index_wm, lat, lon FROM {table_weathermodel} WHERE {name_of_index_wm} in ({ix_wm})"

    df_coords = pd.read_sql(query,engine)
    
    return df_coords



def write_weatherdata_into_db(df_weatherdata,tablename_weatherdata,engine):
    

    print(df_weatherdata.head())

    df_weatherdata.to_sql(index_label='time', name=tablename_weatherdata, con=engine,schema='public', if_exists='append')




def mark_ts_entries_with_wm_index(entrys_wo_weather,dataset_ts,dataset_id,dataset_time,dataset_ix_wm,engine):
    """
    give an index entry to those id and timesteps, for which are weatherdata in db
    """


    for i,row in entrys_wo_weather.iterrows():

        id = row['id']
        index_wm = row['index_wm']
        min_time = row['mintime']
        max_time = row['maxtime']

        
        query = f"""UPDATE {dataset_ts} as data 
                SET {dataset_ix_wm} = {index_wm}
                WHERE data.{dataset_id} = '{id}' AND {dataset_time} >= '{min_time}' AND {dataset_time} <= '{max_time}'
                """
        
        print(query)

        with engine.connect() as connection:
            connection.execute(text(query))
            connection.commit()  # commit the transaction
        




if __name__ == '__main__':

    print('start')
    ## Variables wpuq
    db_name_data = 'last'
    db_name_weather = 'last'
    dataset_ts = 'wpuq_ts'
    dataset_meta = 'wpuq_meta'
    dataset_id = 'id'
    dataset_time_column = 'time'
    dataset_ix_wm = 'id_era5_land'
    table_weathermodel = 'era5_land_centroid_deu_landpixel'
    table_wm_index = 'id_era5_land'
    tablename_weatherdata = 'era5_land'
    name_geom_column = 'geom'


    # Verbindung DB Lastdaten
    engine_data = get_engine(db_name_data)

    # Verbindung DB Lastdaten
    if db_name_weather == db_name_data:
         engine_weather = engine_data
    else:
        engine_weather = get_engine(db_name_weather)


    ## 0.1 Lookuptable der Wettermodellcentroide in die Datenbank schreiben
    # filepath=r"C:\Users\agoldmaier\Documents\meteoData\ERA5_Land_grid.nc"
    # write_centroids_in_db(filepath,table_weathermodel,engine_data)
    # print('0.1 done')

    ## 0.2 Geometriespalte in Metadata hinzufügen
    # add_geom_column(tablename=dataset_meta,engine=engine_data,name_geom=name_geom_column)
    # print('0.2 done')
    # name_geom_column = 'geom_plz'

    ## 1. Wetterdatenindex zu Metadaten hinzufügen, wenn noch nicht vorhanden
    add_index_weatherdata(dataset_meta,dataset_id,dataset_ix_wm,table_weathermodel,table_wm_index,engine_data,column_geom=name_geom_column)
    print('1. done')

    ## 2. Tabelle für Wetterdaten erzeugen, wenn noch nicht vorhanden
    create_table_weatherdata(tablename_weatherdata,engine_weather)
    print('2. done')

    ## 3. In Zeitreihendaten prüfen, für welche Orte und Zeitstempel noch keine Wetterdaten zugeordnet sind.
    entrys_wo_weather = find_entrys_wo_weatherdata(dataset_ts,dataset_meta,dataset_id,dataset_time_column,dataset_ix_wm,engine_data)
    print('3. done')
    print(entrys_wo_weather.head())

    ## 4. In Wetterdatentabelle prüfen, welche der fehlenden Zeiträume vorhanden sind und welche nicht. 
    missingYears = find_missing_years_and_fill(entrys_wo_weather,engine_weather)
    print('4. done')
    print(missingYears)

    if not missingYears.empty:
        missingYears_unique = missingYears[['index_wm','Missing Years']].drop_duplicates()

        ## 5. Koordinaten der Wettermodelpixel abfragen
        coords_wm = get_coords_weathermodel(missingYears_unique['index_wm'],table_weathermodel,table_wm_index,engine_weather)
        print('5. done')
        missingYears_unique = missingYears_unique.merge(coords_wm, left_on='index_wm', right_on='index_wm')

        print(missingYears_unique)

        ## 6. Die fehlenden Wetterdaten aus dem Wettermodell abfragen
        wd_missing = get_weatherdata_from_model(missingYears_unique)
        print('6. done')
        print(wd_missing.head())

        ## 7. Die Wetterdaten in die Datenbank schreiben
        write_weatherdata_into_db(wd_missing,tablename_weatherdata,engine_weather)
        print('7. done')


    # Fällt weg wegen Performance (Datenbank stürzt ab, bei der Menge der Anfragen)
    ## 8. Alle Einträge aus 3. mit index aus Wettermodell versehen, damit klar ist, dass die Wetterdaten vorhanden sind.
    # mark_ts_entries_with_wm_index(entrys_wo_weather,dataset_ts,dataset_id,dataset_time_column,dataset_ix_wm,engine_data)
    # print('8. done')

    # Datenbankverbindung schließen
    engine_data.dispose()
    engine_weather.dispose()
    print('finish')



    # lat_list = [50.57,50.57,51.06,51.34,51.33,51.18,48.04,51.23,51.40,51.09,51.05]
    # lon_list = [12.53,12.53,13.67,12.40,12.40,11.91,10.73,13.13,12.32,13.72,13.77]
    # index_wm_list = ['401ba297-e962-4e75-97c9-7f7f20fb48ca','bfc084c6-3b55-4bd8-b897-fa360012bf15',
    #                  'da798658-93b2-4247-8b23-66ef0eb46bb6','ce785826-7e58-4582-9b45-048efbe48857',
    #                  'd31632fc-6e04-4b1b-a450-d24076fb23ff','912df36e-fd7c-49bf-a072-40b9b52294d2',
    #                  '4337ad00-30a3-47d9-8f6f-2e7ff6086801','7a906aaf-2609-43d8-bcf5-2c4d73d1d822',
    #                  'c03cbd32-1df6-4655-b73a-4753ef06fb83','3dbb03ca-239f-4133-ab4c-92cb81dae68d',
    #                  'fa05d21a-c315-443a-84b4-c96006bbb8f7']
    # year_list = [2018,2019,2020,2021,2022,2023]

    # wd_missing = get_weatherdata_from_model(lat_list,lon_list,index_wm_list,year_list)
    
    # wd_missing.to_csv(r'C:\Users\agoldmaier\Documents\Projekte_tmp\SyLas-KI\Daten\openMeter\weatherData.csv')



# %%
