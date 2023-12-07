import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# 주파수와 진폭을 사용하여 가상의 진동 데이터 생성
np.random.seed(42)
n_samples = 200
frequency = np.random.uniform(1, 5, n_samples)  # 주파수 (1~5 Hz 범위에서 무작위 선택)
amplitude = np.random.uniform(1, 10, n_samples)  # 진폭 (1~10 범위에서 무작위 선택)

# 가상의 레이블 생성 (임의로 0 또는 1 선택)
labels = np.random.choice([0, 1], size=n_samples)

# 데이터 시각화
plt.scatter(frequency, amplitude, c=labels, cmap=plt.cm.RdYlBu, edgecolor='k')
plt.title("Simulated Vibration Safety Data")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.show()

# 데이터 분할
features = np.column_stack((frequency, amplitude))
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 랜덤 포레스트 분류 모델 훈련
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 모델 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 결과 출력
print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

