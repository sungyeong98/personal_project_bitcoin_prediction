# personal_project_bitcoin_prediction

### 프로젝트 개요

해당 개인 프로젝트는 비트코인 시장에서의 가격 변동을 예측하기 위해 딥러닝 모델을 활용하는 것을 목표로 합니다. 비트코인의 1분봉 데이터를 50만개 수집하여, 각기 다른 딥러닝 아키텍처인 RNN, LSTM, GRU 모델에 데이터를 입력하여 학습시켰습니다. 이를 통해 시계열 데이터에서의 패턴을 학습하고, 다음 시간대의 종가를 예측하는 모델을 개발하였습니다.

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