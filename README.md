# personal_project_bitcoin_prediction

### 프로젝트 개요

인공신경망 모델을 통한 비트코인 주가 예측 모델

<br>

### 기술 스택

<img src="https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white"> <img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white">
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">

<br>

### 사용 모델

RNN, LSTM, GRU

<br>

### 폴더 구조

data/

- 'raw_data': pyupbit을 통해 추출한 1분봉 데이터를 저장한 디렉토리
- 'preprocessing_data': 전처리된 데이터를 저장한 디렉토리

model/

- 'main_model': 프로젝트에 사용된 모델과 실제 예측값을 출력하는 코드가 저장된 디렉토리
- 'save_model': 학습된 모델이 저장된 디렉토리
- 'test_model': 학습에 사용할 모델의 임시 테스트 코드가 저장된 디렉토리

preprocessing/

- pyupbit을 통한 데이터 추출 코드 및 전처리 코드가 저장된 디렉토리