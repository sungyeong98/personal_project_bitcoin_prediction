#raw_data를 전처리하는 코드

import os
import pandas as pd

#raw데이터를 불러오기
script_dir=os.path.dirname(__file__)
relative_path="../data/raw_data/raw_data.csv"
file_path=os.path.join(script_dir,relative_path)

df=pd.read_csv(file_path)
#Unnamed: 0 = 시간정보
#open = 시작가격
#high = 고가
#low = 저가
#close = 종가
#volume = 거래량(거래된 코인의 개수)
#value = 거래금액(거래된 코인의 개수에 대한 가격)

#Unnamed: 0을 time으로 이름변경
df.rename(columns={'Unnamed: 0':'time'},inplace=True)

del_col=['time','value']
df=df.drop(del_col,axis=1)
#가공된 데이터를 저장
relative_path1="../data/preprocessing_data/preprocessing_data.csv"
file_path1=os.path.join(script_dir,relative_path1)
df.to_csv(file_path1,index=False)