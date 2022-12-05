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
ml_train .\configs\cle_hea_diss_simple.json --stream --debug
ml_predict "..\test.csv" ".\models\RandomForest.pkl" --output_path "..\output.csv" --stream
ml_predict "..\test.csv" ".\models\RandomForest.pkl" --stream
~~~

Test:
~~~
python -m unittest discover ml_project/
~~~

Project Organization
------------

    
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── raw            <- The original, immutable data dump.    
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          
    │
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── ml_project                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
         
        ├── data           <- code to download or load data
         
        ├── features       <- code to turn raw data into features for modeling
         
        ├── models         <- code to train models and then use trained models to make
    


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

