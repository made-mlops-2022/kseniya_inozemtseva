from setuptools import setup

setup(
    name="preprocess",
    version='0.1',
    py_modules=['preprocess'],
    install_requires=[
        'Click',
    ],
    entry_points={
        'console_scripts': [
            'preprocess = preprocess:preprocess',
            'split = preprocess:split',
            'train = preprocess:train',
            'validate = preprocess:validate',
        ]
    }
)