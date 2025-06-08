# embedding_telegram

## How to install 

1) Вам необходимо установить [uv](https://github.com/astral-sh/uv) и [python](https://www.python.org/)
2) Скачайте и распакуйте репозиторий на диск
3) Выполните следующие команды: `uv venv`, для macOS/Linux `source .venv/bin/activate`, `uv sync`
4) Для запуска embeding модели я использую lm studio и модель `granite-embedding-278m-multilingual-GGUF/granite-embedding-278m-multilingual-Q8_0.gguf`
4) Запустите `python3 show.py`
5) Откройте в браузере `localhost:8052`
6) Далее вам необходимо экспортировать переписку telegram в "машиночитаемом json" и загрузить ее в браузер
7) После нижимайте `Go to visualization`