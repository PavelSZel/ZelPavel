# Папка содержит следующие файлы:

- BertSentimentClassification - код обучения Bert(Из-за виджетов не отображается на GitHub, но его можно скачать и посмотреть локально)
- BertSentimentClassification2 - код обучения модели с удаленными выходами для ознакомления на GitHub
- SentimentCatBoost - код обучения Catboost
- SentimentAPI - код FastAPI для обработки post-запроса
- model.cbm - сохранённая модель CatBoost
- /model - директория хранения модели Bert

[Данные для обучения моделей](https://www.kaggle.com/datasets/senylar/sis-text-class/data)


В файлах **BertSentimentClassification**, **SentimentCatBoost** мы обучаем модели на данных из kaggle для оценки сентимента отзыва.

Сохраняем модели в **/model** и **model.cbm** соответственно, для загрузки в файл **SentimentAPI**.

Файл **SentimentAPI** содержит в себе код реализации post-запросов к моделям.

Внутри файла есть инструкция как запустить локальный сервис.

Для отправки запроса через **Postman**:

  **POST** -> http://127.0.0.1:8000/predict_cat_boost  (для обращения к модели Catboost)
            
  **POST** -> http://127.0.0.1:8000/predict_bert       (для обращения к модели Bert)
  
  -> Body -> raw -> Text -> Введите текст -> Send
                 


Внутри файла SentimentAPI есть так же реализация локальной проверки без post-запроса. 
Реализация моделей несет в себе ознакомительный характер, модель может ошибаться. На примере из SentimentAPI можно увидеть разницу,
при обращении к разным моделям. 
