#LSTM을 이용한 모델 코드
#====================
#필요한 라이브러리 호출
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential,Model,load_model
from keras.layers import Input,Dense,Activation,Flatten,Dropout
from keras.layers import SimpleRNN,LSTM,GRU
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
print('======================데이터셋=========================')
print('X_train : ',X_train.shape, ' y_train : ',y_train.shape)
print('X_test : ',X_test.shape, ' y_test : ',y_test.shape)
print('======================================================')
#====================
#모델 설정
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, activation='relu'))
model.add(Dropout(0)) 
model.add(LSTM(256, return_sequences=True, activation="relu"))
model.add(Dropout(0)) 
model.add(LSTM(128, return_sequences=True, activation="relu"))
model.add(Dropout(0)) 
model.add(LSTM(64, return_sequences=False, activation="relu"))
model.add(Dropout(0)) 
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
model_fit = model.fit(X_train, y_train, 
                      batch_size=32, epochs=10,
                      verbose=1)
#====================
#모델 평가
result=model.evaluate(X_test,y_test)
y_test_pred=model.predict(X_test)
y_test_pred_aggr=np.mean(y_test_pred,axis=1)
y_test_pred_inverse=scaler_y.inverse_transform(y_test_pred_aggr.reshape(-1,1))

y_test_original=scaler_y.inverse_transform(y_test)
mse=mean_squared_error(y_test_original,y_test_pred_inverse)
mae=mean_absolute_error(y_test_original,y_test_pred_inverse)
r2=r2_score(y_test_original,y_test_pred_inverse)
print('======================모델평가=========================')
print("Test score : ",result)
print(f'Mean Squared error : {mse}')
print(f'Mean Absolute error : {mae}')
print(f'R-squared : {r2}')
print('======================================================')

relative_path1="../save_model/LSTM_model.h5"
file_path1=os.path.join(script_dir,relative_path1)
model.save(file_path1)