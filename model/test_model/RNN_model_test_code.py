#RNN을 사용한 모델의 테스트를 위한 코드이다.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#전처리된 csv파일을 불러오기
script_dir=os.path.dirname(__file__)
relative_path="../../data/preprocessing_data/temporary_preprocessing_data.csv"
file_path=os.path.join(script_dir,relative_path)
df=pd.read_csv(file_path)



from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()


#임시로 학습데이터와 테스트데이터를 7:3으로 분배
n=int(len(df)*0.7)

x_cols=[0,1,2,4]
y_cols=[3]

X=df.iloc[:,x_cols].values
y=df.iloc[:,y_cols].values

scaler_X=MinMaxScaler()
scaler_y=MinMaxScaler()

X_scaled=scaler_X.fit_transform(X)
y_scaled=scaler_y.fit_transform(y)

X_window,y_window=[],[]
for i in range(len(X_scaled)-5+1):
    X_window.append(X_scaled[i:i+5,:])
    y_window.append(y_scaled[i+5-1])
X_window=np.array(X_window)
y_window=np.array(y_window)

X_train,X_test=X_window[:n],X_window[n:]
y_train,y_test=y_window[:n],y_window[n:]
#각 데이터들을 3차원으로 수정
#X_train,y_train=np.array(X_train),np.array(y_train)[:,np.newaxis]   #y의 값은 종가 하나만 들어가서 이걸 그냥 np.array해주면 1차원이됨
#X_test,y_test=np.array(X_test),np.array(y_test)[:,np.newaxis]
print('X_train : ',X_train.shape, ' y_train : ',y_train.shape)
print('X_test : ',X_test.shape, ' y_test : ',y_test.shape)
#=============================================================
#학습에 필요한 라이브러리 호출

from keras.models import Sequential,Model,load_model
from keras.layers import Input,Dense,Activation,Flatten,Dropout
from keras.layers import SimpleRNN,LSTM,GRU
#=============================================================

#RNN
model_rnn=Sequential()

model_rnn.add(SimpleRNN(128,input_shape=(X_train.shape[1],X_train.shape[2]),return_sequences=True,activation='relu'))
model_rnn.add(Dropout(0))
model_rnn.add(SimpleRNN(256,return_sequences=True,activation='relu'))
model_rnn.add(Dropout(0))
model_rnn.add(SimpleRNN(128,return_sequences=True,activation='relu'))
model_rnn.add(Dropout(0))
model_rnn.add(SimpleRNN(64,return_sequences=True,activation='relu'))
model_rnn.add(Dropout(0))
model_rnn.add(Flatten())
model_rnn.compile(optimizer='adam',loss='mean_squared_error')
model_rnn.summary()
model_rnn_fit=model_rnn.fit(X_train,y_train,batch_size=8,epochs=10,verbose=1)


#모델 평가
result=model_rnn.evaluate(X_test,y_test)
print("Test score : ",result)


y_test_pred=model_rnn.predict(X_test)
print(y_test_pred.shape)

y_test_pred_aggr=np.mean(y_test_pred,axis=1)
print(y_test_pred_aggr.shape)

y_test_pred_inverse=scaler_y.inverse_transform(y_test_pred_aggr.reshape(-1,1))
print(y_test_pred_inverse.shape)

#==============================================
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
y_test_original=scaler_y.inverse_transform(y_test)
#MSE(평균 제곱 오차)
mse=mean_squared_error(y_test_original,y_test_pred_inverse)
print(f'Mean Squared error : {mse}')

#MAE(평균 절대 오차)
mae=mean_absolute_error(y_test_original,y_test_pred_inverse)
print(f'Mean Absolute error : {mae}')

#R-squared(결정 계수)
r2=r2_score(y_test_original,y_test_pred_inverse)
print(f'R-squared : {r2}')