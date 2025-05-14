from fastapi import FastAPI, Request
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import dill
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab, Doc


api = FastAPI()


model = AutoModelForSequenceClassification.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model")
# Загружаем модель Bert
bert = pipeline('text-classification', model=model, tokenizer=tokenizer)


segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()
# Функция для обработки текста
def lemmatize(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    lemmas = [token.lemma.lower() for token in doc.tokens if token.lemma.isalpha()]
    return  ' '.join(lemmas)


# Словарь для корректного отображения категорий логистической регрессией
names_dict = {0: 'athletics',
 1: 'autosport',
 2: 'basketball',
 3: 'boardgames',
 4: 'esport',
 5: 'extreme',
 6: 'football',
 7: 'hockey',
 8: 'martial_arts',
 9: 'motosport',
 10: 'tennis',
 11: 'volleyball',
 12: 'winter_sport'}


with open('LogRegModel.dill', 'rb') as f:
    model = dill.load(f) # Модель логистической регрессии


# post-запрос к логистической регрессии
@api.post('/predict_lr')
async def predict(text: Request):
    body = await text.body()
    raw = body.decode('utf-8')

    lemmas = lemmatize(raw)
    pred = model.predict([lemmas])[0]

    return names_dict[int(pred)]


# post-запрос к bert
@api.post('/predict_bert')
async def predict_bert(text: Request):
    body = await text.body()
    raw = body.decode('utf-8')

    return bert(raw)


### Для запуска локального хоста выполнить в командной строке:  uvicorn SportClassification:api

### Примеры для проверки ответа модели:

# Медведев закончил игру подачей навылет. Гейм, сет, матч
# Отличный прыжок нашего спортсмена со снаряда, идеальное приземление
