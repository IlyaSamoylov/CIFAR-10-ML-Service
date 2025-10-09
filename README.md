# Тестовое Задание: Разработка ML-Сервиса для Классификации Изображений на FastAPI (CIFAR-10)
## Быстрый старт:
## Quick start

1. Склонировать репозиторий:
```bash
git clone https://github.com/<username>/<repo>.git
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
