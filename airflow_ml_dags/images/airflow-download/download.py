import os
from random import randint
import click
import numpy as np
from sklearn.datasets import make_classification
from sklearn.utils import shuffle
from datetime import date
import logging



@click.command("download")
@click.argument("output_dir")
def download(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger('Downloader')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(output_dir, 'dowloading.log'))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.debug("here we are")
    d = dict(
        n_samples=randint(80, 120),
        n_features=20,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        weights=[0.2, 0.8],
        flip_y=0.1,
        class_sep=1.0,
        hypercube=True,
        shift=np.arange(-10, 10),
        scale=np.arange(1, 21),
        shuffle=False
    )

    X, y = make_classification(**d)

    #np.savetxt(os.path.join(output_dir, f"data_{date.today().strftime('%d%m%Y')}.csv"), X)
    #np.savetxt(os.path.join(output_dir, f"target_{date.today().strftime('%d%m%Y')}.csv"), y)
    np.savetxt(os.path.join(output_dir, f"data.csv"), X, delimiter=',')
    np.savetxt(os.path.join(output_dir, f"target.csv"), y, delimiter=',')


if __name__ == '__main__':
    download()
