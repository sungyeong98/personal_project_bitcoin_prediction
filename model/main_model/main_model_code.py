#메인 모델을 이용한 코드
#=========================================
#라이브러리 호출
import os
import pandas as pd
import numpy as np
import pyupbit
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime
#=========================================
#예측에 필요한 데이터 추출
bitcoin=pyupbit.get_ohlcv('KRW-BTC',interval='minute1',count=20)

del_col=['value']
bitcoin=bitcoin.drop(del_col,axis=1)
#=========================================
#데이터 가공
scaler=MinMaxScaler()
x_cols,y_cols=[0,1,2,4],[3]
X,y=bitcoin.iloc[:,x_cols].values,bitcoin.iloc[:,y_cols].values
scaler_X,scaler_y=MinMaxScaler(),MinMaxScaler()
X_scaled,y_scaled=scaler_X.fit_transform(X),scaler_y.fit_transform(y)

X_window,y_window=[],[]
for i in range(len(X_scaled)-5+1):
    X_window.append(X_scaled[i:i+5,:])
    y_window.append(y_scaled[i+5-1])
X_input,y_input=np.array(X_window),np.array(y_window)
#=========================================
#모델 불러오기
script_dir=os.path.dirname(__file__)
relative_path1="../save_model/RNN_model.h5"
file_path1=os.path.join(script_dir,relative_path1)
model=load_model(file_path1)
#=========================================
#값 예측
y_pred=model.predict(X_input)
y_pred_aggr=np.mean(y_pred,axis=1)
y_pred_inverse=scaler_y.inverse_transform(y_pred_aggr.reshape(-1,1))
y_original=scaler_y.inverse_transform(y_input)
#=========================================
#결과값 그래프화
'''
plt.figure(figsize=(10, 6))
plt.plot(y_original, label='Actual', marker='o',linewidth=1.5)
plt.plot(y_pred_inverse, label='Predicted', marker='x',linewidth=1.5)
plt.title('Actual vs Predicted')
plt.xlabel('Data Point')
plt.ylabel('Value')
plt.legend()
plt.show()
'''
#=========================================
#결과값 출력
result="{:,.0f}원".format(y_pred_inverse[-1][0])
current_time=datetime.now()
year,month,day,hour,minute=current_time.year,current_time.month,current_time.day,current_time.hour,current_time.minute
print(f'{year}년 {month}월 {day}일 {hour}시 {minute}분 종가는 {result}로 예상됩니다.')