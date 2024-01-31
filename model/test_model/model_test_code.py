#LSTM을 사용한 모델의 테스트를 위한 코드이다.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#전처리된 csv파일을 불러오기
script_dir=os.path.dirname(__file__)
relative_path="../../data/preprocessing_data/temporary_preprocessing_data.csv"
file_path=os.path.join(script_dir,relative_path)
df=pd.read_csv(file_path)


#임시로 학습데이터와 테스트데이터를 7:3으로 분배
n=int(len(df)*0.7)
train_data,test_data=df.iloc[:n],df.iloc[n:]

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

train_scaled=scaler.fit_transform(train_data)
test_scaled=scaler.transform(test_data)

#X에는 시작가,고가,저가,거래량을 넣어줌
#y에는 종가를 넣어줌
X_train,y_train=[],[]
for i in range(len(train_scaled)-5):
    X_train.append(train_scaled[i:i+5][:,[0,1,2,4]])
    y_train.append(train_scaled[i+5,3])

X_test,y_test=[],[]
for i in range(len(test_scaled)-5):
    X_test.append(test_scaled[i:i+5][:,[0,1,2,4]])
    y_test.append(test_scaled[i+5,3])

#각 데이터들을 3차원으로 수정
X_train,y_train=np.array(X_train),np.array(y_train)[:,np.newaxis]   #y의 값은 종가 하나만 들어가서 이걸 그냥 np.array해주면 1차원이됨
X_test,y_test=np.array(X_test),np.array(y_test)[:,np.newaxis]

#=============================================================
#학습에 필요한 라이브러리 호출

from keras.models import Sequential,Model,load_model
from keras.layers import Input,Dense,Activation,Flatten,Dropout
from keras.layers import SimpleRNN,LSTM,GRU
#=============================================================

#RNN
model_rnn=Sequential()

model_rnn.add(SimpleRNN(128,input_shape=(X_train.shape[1],X_train.shape[2]),return_sequences=True,activation='relu'))
model_rnn.add(Dropout(dropout_ratio))
model_rnn.add(SimpleRNN(256,return_sequences=True,Activation='relu'))
model_rnn.add(Dropout(dropout_ratio))
model_rnn.add(SimpleRNN(128,return_sequences=True,Activation='relu'))
model_rnn.add(Dropout(dropout_ratio))
model_rnn.add(SimpleRNN(64,return_sequences=True,Activation='relu'))
model_rnn.add(Dropout(dropout_ratio))
model_rnn.add(Flatten())
model_rnn.compile(optimizer='adam',loss='mean-squared-error')
model_rnn.summary()
model_rnn_fit=model_rnn.fit(X_train,y_train,batch_size=8,epochs=10,verbose=1)

y_train_pred=model_rnn_fit.predict(X_train)
y_test_pred=model_rnn_fit.predict(X_test)

