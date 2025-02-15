import pandas as pd
import tensorflow as tf
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv('../Dataset_cleaned.csv')
print(data)
data['content'] = data['content'].str.replace('[^ㄱ-ㅎ ㅏ-ㅣ 가-힣 ]','',regex=True)
print(data['content'].isna().sum()) # null 값 있는지 확인


#토큰 설정 OOV => 임시 데이터 설정정
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<OOV>')


data_set = data['content'].tolist()
labels = data['label'].tolist()  
# 토큰화  (단어:숫자자)
tokenizer.fit_on_texts(data_set)

# print(tokenizer.word_index)

# 각단어 숫자 세기기
word_counts = Counter()
for text in data_set:
    for word in text.split():
        word_counts[word] += 1


original_vocab_size = len(tokenizer.word_index) + 1  # OOV 포함

filtered_words = {'<OOV>': 1}  # OOV를 첫 번째 인덱스로 유지
filtered_words.update({word: index + 1 for index, (word, count) in enumerate(tokenizer.word_index.items()) if word_counts[word] >= 2})

tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<OOV>')
tokenizer.word_index = filtered_words

import pickle

# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)