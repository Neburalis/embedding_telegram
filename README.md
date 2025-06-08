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
7) После нижимайте `Go to visualization`