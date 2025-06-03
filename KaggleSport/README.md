# Проект состоит из:

- /data - [данные для обучения](https://www.kaggle.com/datasets/mikhailma/russian-social-media-text-classification/data)
- LogReg_and_RandFor_SportClassification - Обучение моделей логистической регрессии рандомного леса с помощью библиотеки natasha
- LogRegModel.dill - сохранённая модель логистической регрессии(векторизатор + модель)
- BertSportClassification - обучение модели bert для локального ознакомления. GitHub не показывает файл из-за виджетов. 
- BertSportClassification.2 - файл обучения без выходов ячеек, для ознакомления с процессом на GitHub
- /model - сохранённая модель bert
- SportClassification.py - реализация FastApi для двух моделей


Файл с реализацией Fastapi содержит инструкцию по запуску локального хоста и примеры ввода. 

Данные отправлять в формате Body -> raw -> Text

Для обращения к модели логистической регрессии: **POST** -> http://127.0.0.1:8001/predict_lr

Для обращения к модели bert: **POST** -> http://127.0.0.1:8001/predict_bert
