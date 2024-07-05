"""Database connection to read and write data"""

from sqlalchemy.engine import Engine, Connection
from sqlalchemy import create_engine, engine
from pathlib import Path
import configparser
from syndatagenerators.data_preparation import dload_wpuq, dload_snh, dload_london, dload_openmeter, dload_chargingstation

class DBHandler:
    def __init__(self, dataset):
        self.dataset = dataset
        
        config = configparser.ConfigParser()
        config.read(Path("../data_preparation/last_db.ini"))
        
        configs = {
            "wpuq": "last",
            "snh": "last_secure",
            "d4g": "last_secure",
            "lsm": "last",
            "openmeter": "last",
            "chargingstation": "last_secure"
        }
        pgsql_info = dict(config[configs[dataset]])
        self._connect(pgsql_info)
    
    def _connect(self, pgsql_info):
        try:
            self.engine = create_engine(f"postgresql://{pgsql_info['user']}:"
                                    f"{pgsql_info['password']}"
                                    f"@{pgsql_info['host']}/{pgsql_info['database']}")
        except Exception as e:
            print(f'Verbindung zur DB nicht möglich:{e}')
    
    def get_data(self, **params):
        dataloader = {
            "wpuq": dload_wpuq.get_ts_wpuq,
            "snh": dload_snh.get_ts_snh,
            "d4g": lambda **params: dload_snh.get_ts_snh(**params, tablename="smartmeter_data4grid_snh"),
            "lsm": dload_london.get_ts_loandonsmartmeter,
            "openmeter": dload_openmeter.get_ts_openmeter,
            "chargingstation": dload_chargingstation.get_ts
        }
        return dataloader[self.dataset](**params, engine=self.engine)
    
    def get_metadata(self):
        dataloader = {
            "wpuq": dload_wpuq.get_meta_wpuq,
            "snh": lambda engine: dload_snh.get_meta_snh(engine=engine, tablename="smartmeter_snh_meta"),
            "d4g": lambda engine: dload_snh.get_meta_snh(engine=engine, tablename="smartmeter_data4grid_snh_meta"),
            "lsm": None,  # no metadata for LSM dataset
            "openmeter": dload_openmeter.get_meta_openmeter,
            "chargingstation": None # no metadata
        }
        result = dataloader[self.dataset](self.engine)
        return result


class LondonSmartMeter:
    """
    Handle connection to database for loading the London Smart Meter data.
    """

    def __init__(self, user: str, password: str):
        self.connection: Connection = None
        self.engine: Engine = None

        self.host = r'mydbhost.com'
        self.port = 5423
        self.database = 'dbname'
        self._connection_string: str = 'postgresql://{user}:{password}@{host}/{db}'.format(
            user=user,
            password=password,
            host=self.host,
            db=self.database
        )

    def connect(self) -> None:
        self.engine = create_engine(self._connection_string)
        self.connection = self.engine.connect()
