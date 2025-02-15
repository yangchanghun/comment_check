import pandas
from googleapiclient.discovery import build
from multiprocessing.dummy import Pool as ThreadPool

url_id = ['ooWegesHKwY']

def croll_youtube(i):
    api_key = 'api키를 입력하시오오'
    video_id = i
    comments = list()
    api_obj = build('youtube', 'v3', developerKey=api_key)
    response = api_obj.commentThreads().list(part='snippet,replies', videoId=video_id, maxResults=100).execute()
    while response:
        for item in response['items']:
            if len(comments) >200:
                break
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append([comment['textDisplay'], comment['authorDisplayName'], comment['publishedAt'], comment['likeCount']])
    
            if item['snippet']['totalReplyCount'] > 0:
                for reply_item in item['replies']['comments']:
                    reply = reply_item['snippet']
                    comments.append([reply['textDisplay'], reply['authorDisplayName'], reply['publishedAt'], reply['likeCount']])
    
        if 'nextPageToken' in response:
            response = api_obj.commentThreads().list(part='snippet,replies', videoId=video_id, pageToken=response['nextPageToken'], maxResults=100).execute()
        else:
            break
    df = pandas.DataFrame(comments)
    df[0] = df[0].str.replace('[^ㄱ-ㅎ ㅏ-ㅣ 가-힣 ]','',regex=True)
    print(df[0])
    data_set = df[0].tolist()
    return data_set





pool = ThreadPool(12)

data_set = pool.map(croll_youtube,url_id)
pool.close()
pool.join()



from tensorflow.keras.models import load_model
import tensorflow as tf
import pickle
model = load_model('best_performed_model.h5')
print("✅ 전체 모델을 성공적으로 불러왔습니다!")
# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_text(model, tokenizer, text, maxlen=100):
    # 1️⃣ 입력 문장을 토큰화하여 시퀀스로 변환
    sequence = tokenizer.texts_to_sequences([text])

    # 2️⃣ 패딩 적용 (입력 길이를 맞추기 위해)
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=maxlen)

    # 3️⃣ 예측 수행
    prediction = model.predict(padded_sequence)

    # 4️⃣ 확률을 0 또는 1로 변환
    label = (prediction > 0.5).astype(int)[0][0]

    # 5️⃣ 결과 출력
    print(f"🔹 입력 문장: {text}")
    print(f"🔹 예측된 라벨: {label} (0: 부정, 1: 긍정)")
    print(f"🔹 예측된 predict: {prediction} (0: 부정, 1: 긍정)")


from tensorflow.keras.models import load_model

# 저장된 모델 불러오기
model = load_model("best_performed_model.h5")

# 새로운 문장 예측 예제
sample_text = data_set[0]
for i in sample_text:
    predict_text(model, tokenizer, i)