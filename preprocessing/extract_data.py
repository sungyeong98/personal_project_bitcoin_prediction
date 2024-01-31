#비트코인 1분봉 데이터를 가져와서 csv파일로 변환하여 저장
#pip install -r requirements.txt

import pyupbit
import pandas as pd
import os
#print(pyupbit.get_ohlcv("KRW-BTC", interval="day", count=5))    # 일봉 데이터 (5일)
#print(pyupbit.get_ohlcv("KRW-BTC", interval="minute1",count=10))         # 분봉 데이터
#print(pyupbit.get_ohlcv("KRW-BTC", interval="week"))            # 주봉 데이터

bitcoin_rawdata=pyupbit.get_ohlcv('KRW-BTC',interval='minute1',count=500000)

script_dir=os.path.dirname(__file__)
relative_path="../data/raw_data/raw_data.csv"
file_path=os.path.join(script_dir,relative_path)

bitcoin_rawdata.to_csv(file_path)