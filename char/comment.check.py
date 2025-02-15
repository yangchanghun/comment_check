import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Embedding, GlobalMaxPooling1D
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf



data = pd.read_csv('../Dataset_cleaned.csv')
data['content'] = data['content'].str.replace('[^ㄱ-ㅎ ㅏ-ㅣ 가-힣 ]','',regex=True)
label_counts = data["label"].value_counts(normalize=True) * 100
invalid_label_rows = data[~data['label'].astype(str).isin(['0', '1'])]  # 문자열로 변환 후 필터링
#토큰화한거 가지고 ㅗㅇ기기
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)



data_shuffled = data.sample(frac=1, random_state=777).reset_index(drop=True)

# 🔹 텍스트와 라벨 분할
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    data_shuffled['content'], data_shuffled['label'], test_size=0.3, random_state=777
)  # 70% train

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=2/3, random_state=777
)  # 10% val, 20% test

# 🔹 시퀀스로 변환 (각 데이터셋 별로 처리)
train_sequences = tokenizer.texts_to_sequences(train_texts)
val_sequences = tokenizer.texts_to_sequences(val_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# 🔹 최대 시퀀스 길이 설정 (예: 100)
maxlen = 100

# 🔹 패딩 적용
x_train = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=maxlen)
x_val = tf.keras.preprocessing.sequence.pad_sequences(val_sequences, maxlen=maxlen)
x_test = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=maxlen)

# 🔹 라벨을 NumPy 배열로 변환
y_train = np.array(train_labels)
y_val = np.array(val_labels)
y_test = np.array(test_labels)
print(f"y_train shape: {y_train.shape}")
print(f"y_train[:5]: {y_train[:5]}")


print(f"Train shape: {x_train.shape}, Validation shape: {x_val.shape}, Test shape: {x_test.shape}")
print(f"Train Labels shape: {y_train.shape}, Validation Labels shape: {y_val.shape}, Test Labels shape: {y_test.shape}")
print(tokenizer.word_index)  # 결과 확인



# ✅ 데이터 변환 (float32로 변환)
x_train = np.array(x_train, dtype=np.int32)
x_val = np.array(x_val, dtype=np.int32)
y_train = np.array(y_train, dtype=np.float32).reshape(-1, 1)
y_val = np.array(y_val, dtype=np.float32).reshape(-1, 1)

# ✅ 모델 정의 (출력 차원 수정)
# 올바른 vocab_size 설정
vocab_size = max(x_train.max(), x_val.max()) + 1

embedding_dim = 16
maxlen = 100


model = tf.keras.models.Sequential([
    Embedding(vocab_size, embedding_dim, input_length=maxlen),
    Bidirectional(LSTM(100, return_sequences=True)),
    GlobalMaxPooling1D(),  # ✅ LSTM의 출력 차원 줄이기
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # ✅ 이진 분류이므로 sigmoid 사용
])

# ✅ 체크포인트 설정
checkpoint_path = 'best_performed_model.weights.h5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, 
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    verbose=1
)

# ✅ 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# ✅ 모델 학습
history = model.fit(
    x_train, y_train, 
    validation_data=(x_val, y_val),
    callbacks=[checkpoint],
    epochs=20, 
    verbose=2
)
model.save("best_performed_model.h5")  # ✅ 모델 전체 저장
m_pred = model.predict(x_test)  # 🔹 테스트 데이터에 대한 예측값
pred = (m_pred > 0.5).astype(int)  # 🔹 확률을 0 또는 1로 변환
true = y_test  # 실제 라벨

# ✅ 예측 결과 출력
print("✅ 예측 결과 확인")
print(f"True labels: {true[:10].flatten()}")  # 🔹 실제 라벨 출력
print(f"Predicted labels: {pred[:10].flatten()}")  # 🔹 예측된 라벨 출력