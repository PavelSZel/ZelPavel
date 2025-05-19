import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io


api = FastAPI()


with open("class_to_idx.json") as f:
    class_to_idx = json.load(f)
# Загружаем словарь соответсвия классов
idx_to_class = {v: k for k, v in class_to_idx.items()}


# Воспроизводим архитектуру модели, как на обучении
model = models.resnet18(weights=None) # Загружаем модель
model.fc = nn.Linear(model.fc.in_features, 2) # Меняем последний слой
model.load_state_dict(torch.load("best_model.pth", map_location="cpu", weights_only=True)) # Подргружаем обученные веса
model.eval()

# Преобразование для изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


@api.post("/predict")                                         # чек-поинт для функции
async def predict_image(file: UploadFile = File(...)):        # функция предсказания для изображения
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        pred_idx = torch.argmax(output, dim=1).item()

    return idx_to_class[pred_idx]  # Возвращаем имя предсказанного класса


### Для запуска хоста:   uvicorn ImageAPI:api
### Для получения предсказания: curl.exe -X POST http://127.0.0.1:8000/predict -F "file=@ИМЯ_ФАЙЛА"
# Файл можно взять любой из /dataset

