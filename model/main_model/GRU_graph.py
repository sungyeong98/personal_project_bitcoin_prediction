#GRU 실제값,예측값의 시각화를 위한 코드
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import matplotlib.pyplot as plt
#====================
#데이터 불러오기
script_dir=os.path.dirname(__file__)
relative_path="../../data/preprocessing_data/preprocessing_data.csv"
file_path=os.path.join(script_dir,relative_path)
df=pd.read_csv(file_path)
#====================
#데이터 스케일링
scaler=MinMaxScaler()
x_cols,y_cols=[0,1,2,4],[3]
X,y=df.iloc[:,x_cols].values,df.iloc[:,y_cols].values
scaler_X,scaler_y=MinMaxScaler(),MinMaxScaler()
X_scaled,y_scaled=scaler_X.fit_transform(X),scaler_y.fit_transform(y)

X_window,y_window=[],[]
for i in range(len(X_scaled)-5+1):
    X_window.append(X_scaled[i:i+5,:])
    y_window.append(y_scaled[i+5-1])
X_window,y_window=np.array(X_window),np.array(y_window)

n=int(len(X_window)*0.7)
X_train,X_test=X_window[:n],X_window[n:]
y_train,y_test=y_window[:n],y_window[n:]
#===============================
#그래프화
relative_path1="../save_model/GRU_model.h5"
file_path1=os.path.join(script_dir,relative_path1)
model=load_model(file_path1)

y_test_pred=model.predict(X_test)
y_test_pred_aggr=np.mean(y_test_pred,axis=1)
#예측값
y_test_pred_inverse=scaler_y.inverse_transform(y_test_pred_aggr.reshape(-1,1))
#실제값
y_test_original=scaler_y.inverse_transform(y_test)

plt.figure(figsize=(10, 6))
plt.plot(y_test_original, label='Actual', marker='o',linewidth=1.5)
plt.plot(y_test_pred_inverse, label='Predicted', marker='x',linewidth=1.5)
plt.title('Actual vs Predicted')
plt.xlabel('Data Point')
plt.ylabel('Value')
plt.legend()
plt.show()