# embedding_telegram

## Project Description

This project is a visualization tool for Telegram chat messages that uses machine learning to analyze and cluster conversations. It processes your Telegram chat export and creates an interactive visualization where similar messages are grouped together.

Key features:
- Processes Telegram chat exports in JSON format
- Uses the Granite embedding model to convert messages into vector representations
- Visualizes message embeddings in 2D space using t-SNE
- Provides interactive clustering using DBSCAN or K-means algorithms
- Allows real-time adjustment of clustering parameters
- Shows message content on hover
- Option to generate new embeddings or use existing ones

## Screenshot

![Visualization Example](screenshot.png)

## How to install 

1) Вам необходимо установить [uv](https://github.com/astral-sh/uv) и [python](https://www.python.org/)
2) Скачайте и распакуйте репозиторий на диск
3) Выполните следующие команды: `uv venv`, для macOS/Linux `source .venv/bin/activate`, `uv sync`
4) Для запуска embeding модели я использую lm studio и модель `granite-embedding-278m-multilingual-GGUF/granite-embedding-278m-multilingual-Q8_0.gguf`
4) Запустите `python3 show.py`
5) Откройте в браузере `localhost:8052`
6) Далее вам необходимо экспортировать переписку telegram в "машиночитаемом json" и загрузить ее в браузер
7) После нажимайте `Go to visualization`
8) В верхней части есть параметры кластеризации, ниже есть их описание 

## Опции генерации эмбедингов

При загрузке файла у вас есть два варианта:
1. **Generate new embeddings** - генерирует новые эмбединги для загруженного файла (может занять некоторое время)
2. **Use existing embeddings.json** - использует уже существующий файл embeddings.json (если он есть в директории проекта)

Если вы выбрали второй вариант, убедитесь что файл embeddings.json существует в директории проекта.

## Алгоритмы кластеризации

В приложении доступны два алгоритма кластеризации:

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

DBSCAN группирует точки, которые находятся близко друг к другу, и помечает как шум точки, которые находятся в областях с низкой плотностью.

Параметры:
- **EPS (Epsilon)** - максимальное расстояние между двумя точками, чтобы они считались соседями. 
  - Меньшие значения создают больше маленьких кластеров
  - Большие значения создают меньше, но более крупных кластеров
  - Диапазон: 0.1 - 3.0
- **Min Samples** - минимальное количество точек, необходимое для формирования кластера
  - Меньшие значения создают больше кластеров
  - Большие значения создают более устойчивые кластеры
  - Диапазон: 1 - 20

### K-means

K-means делит данные на K кластеров, где каждая точка относится к кластеру с ближайшим средним значением.

Параметры:
- **Number of Clusters** - количество кластеров, на которые нужно разделить данные
  - Меньшие значения создают более общие группы
  - Большие значения создают более специфичные группы
  - Диапазон: 2 - 200

### Рекомендации по выбору алгоритма

- Используйте **DBSCAN**, если:
  - Вы не знаете заранее количество кластеров
  - Хотите найти кластеры произвольной формы
  - Хотите автоматически определять выбросы (шум)

- Используйте **K-means**, если:
  - Вы знаете примерное количество кластеров
  - Ожидаете, что кластеры будут примерно одинакового размера
  - Хотите более предсказуемый результат
