import tensorflow as tf
import pandas as pd
import numpy as np

# csv 읽기 ====================================================================================================================================================================================
data = pd.read_csv("gpascore.csv")

# print(data.isnull().sum())  # 빈값  확인
data.dropna()       # 빈값  제거 
# print(data.isnull().sum())  # 빈값 잘 제거되었는지 확인
# exit()

yValues = data["admit"].values
xValues = []
for i, rows in data.iterrows():
    xValues.append([rows["gre"], rows["gpa"], rows["rank"]])

# print(yValues)
# print(xValues)
# exit()

# 딥러닝 모델 만들기 ====================================================================================================================================================================================
# 레이어 만들 때 2의 제곱수로 보통 많이 하더라
# activation = 활성함수. 비선형적 예측을 하고 싶을 때 활성함수를 사용해야 함. 활성함수 사용안하면 선형적 예측
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation="tanh"),
    tf.keras.layers.Dense(128, activation="tanh"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# 딥러닝 모델 컴파일
# optimizer = 경사하강법으로 w를 수정하는데, 새로운 w를 만들 때 최적의 값으로 조정해줌
# loss = 손실함수. binary_crossentropy는 결과가 0과 1 사이의 확률 문제에서 사용
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 딥러닝 모델 학습시키기 (w 최적값 찾기)
learningCnt = 300
model.fit(np.array(xValues), np.array(yValues), epochs=learningCnt)

# 예측
predictValue = model.predict([[750, 3.70, 3], [400, 2.2, 1]])
print(predictValue)