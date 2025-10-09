# Тестовое Задание: Разработка ML-Сервиса для Классификации Изображений на FastAPI (CIFAR-10)
## Быстрый старт:
## Quick start

## Запуск докер контейнера:
1. Скачать образ
```bash
docker pull ghcr.io/ilyasamoylov/cifar-10-ml-service:latest
```
2. Запустить докер контейнер
```bash
docker run -p 8000:8000 ghcr.io/ilyasamoylov/cifar-10-ml-service:latest
```

## Склонировать репозиторий:
1. Клонирование репозитория
```bash
git clone https://github.com/IlyaSamoylov/CIFAR-10-ML-Service
cd <repo>
```
2. Установить зависимости:
```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```
3. Запустить FastAPI:
```bash
uvicorn app.main:app --reload
```

4. Открыть http://127.0.0.1:8000

## Что в репозитории

- `app/` — FastAPI приложение
  - `main.py` — основной сервер (эндпойнты для инференса)
  - `models/` — обученные веса (cnn.pth, resnet18.pth) 
  - `templates/` — HTML шаблоны
  - `static/` — файлы стилей css

- `notebooks/` — Colab ноутбук: EDA - обучение моделей
- `requirements.txt` — зависимости

## Датасет
датасет [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), содержит 60000 изображений 32х32, принадлежащих одному из 10 классов: 
- airplane										
- automobile										
- bird										
- cat										
- deer										
- dog										
- frog										
- horse										
- ship										
- truck
- 
## Как пользоваться приложением
По адресу http://127.0.0.1:8000 выбрать из предложенных моделей, загрузить изображение (можно несколько), нажать загрузить. На открывшейся странице /predict будут json, включающие название файла, выбранная модель для предсказания, индекс и метка предсказанного моделью класса

## Описание моделей
### CNN
Сначала идет 3 блока conv→BN→ReLU→pool, затем два линейных слоя, между ними - dropout

## ResNET18
Изначально используются веса ResNet18_Weights.IMAGENET1K_V1, первый слой заменяется на сверточный, последний линейный слой подстроен для определения 10 классов, во время обучения первый слой был заморожен

## Метрики
### CNN 
Изменение loss'а на тренировочной и валидационной выборке:
<img width="572" height="455" alt="image" src="https://github.com/user-attachments/assets/b63b2a0c-96c7-4805-ba3f-e32d45d27711" />

На тестовой выборке была построена матрица ошибок:
<img width="536" height="478" alt="image" src="https://github.com/user-attachments/assets/23cdb603-fa45-4e08-97af-c6a086f71d2a" />
а также измерены
- точность = 0.81
- f1_macro = 0.81
- f1 для каждого класса = {"plane": 0.82528621, "car": 0.91201609, "bird": 0.71065441, "cat": 0.63533058, "deer": 0.77868459, "dog": 0.70296548, "frog": 0.85742673, "horse": 0.84635018, "ship": 0.90089197, "truck": 0.88356493}

### ResNET18 
Изменение loss'а на тренировочной и валидационной выборке:
<img width="547" height="413" alt="image" src="https://github.com/user-attachments/assets/d382e0cf-9e7f-40f9-9ec2-6ea98718fae7" />

На тестовой выборке была построена матрица ошибок:
<img width="536" height="478" alt="image" src="https://github.com/user-attachments/assets/cb68b74a-d91b-4723-bac1-7b0fc32818c5" />

а также измерены
- точность = 0.9
- f1_macro = 0.89
- f1 для каждого класса = {"plane": 0.90108803, "car": 0.94864048, "bird": 0.86437995, "cat": 0.79573171, "deer": 0.9, "dog": 0.82436261, "frog": 0.92415871, "horse": 0.8987013, "ship": 0.93654697, "truck": 0.93143976}

