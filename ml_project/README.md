ml_example
==============================

Example of ml project

Installation: 
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~
Usage:
Запуск обучения модели командой ml_train после которой надо указать файл конфигурации для запуска. Можно указать флаг --stream для вывода лога в консоль, и флаг --debug для более подробного логгирования (более подробное пишется только в файл с логом).

Запуск обучения - командой ml_predict после которой надо указать csv файл с тестовым датасетом и файл в котором сохранена обученная модель (сообщение в логе из предыдущей команды). Дополнительно можно указать путь до файла, куда сохранять предсказания и те же опции логгирования, что и для предыдущей команды. 
~~~
ml_train "./configs/cle_hea_diss_simple.json" --stream --debug
ml_predict "..\test.csv" ".\models\RandomForest.pkl" --output_path "..\output.csv" --stream
ml_predict "..\test.csv" ".\models\RandomForest.pkl" --stream
~~~

Test:
~~~
pytest tests/
~~~

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── ml_example                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- code to download or generate data
    │   │
    │   ├── features       <- code to turn raw data into features for modeling
    │   │
    │   ├── models         <- code to train models and then use trained models to make
    │   │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


DOCKER:
~~~
python setup.py sdist
docker build -t mikhailmar/train_made:v1 
docker run -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} mikhailmar/train_made:v1
~~~