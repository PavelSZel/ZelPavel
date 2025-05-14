from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from catboost import CatBoostClassifier

api = FastAPI()

model_bert = AutoModelForSequenceClassification.from_pretrained('./model')
tokenizer = AutoTokenizer.from_pretrained('./model')
# Модель Bert
bert_predict = pipeline('text-classification', model=model_bert, tokenizer=tokenizer)

# Модель CatBoost
cb = CatBoostClassifier()
cb.load_model('model.cbm')


sentiment_dict = { # Словарь для человеко-читаемого отображения моделью Catboost
    0: "negative",
    1: "neutral",
    2: "positive"
}

@api.post('/predict_bert')
async def predict_bert(text: Request):
    body = await text.body()
    text = body.decode('utf-8')
    result = bert_predict(text)

    return result



@api.post('/predict_cat_boost')
async def predict_cb(text: Request):
    body = await text.body()
    text = body.decode('utf-8')

    # Делаем предсказание
    y_pred = cb.predict([text])[0]

    # Получаем текстовый ответ с помощью словаря соответствий
    return sentiment_dict[y_pred]


### Для запуска локального сервера запустить в консоли     uvicorn SentimentAPI:api

### Для локальной проверки расскомментировать следующие строки

# text = 'Ну и кантора, очень жалею, что связался с ними'

# print(bert_predict(text)) # CatBoost
# print(sentiment_dict[cb.predict([text])[0]]) # Bert