from setuptools import find_packages, setup


with open('./requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name="ml_project",
    packages=find_packages(),
    version="0.1.0",
    description="Example of ml project",
    author="Your name (or your organization/company/team)",
    entry_points={
        "console_scripts": [
            "ml_train = ml_project.main:launch_train",
            "ml_predict = ml_project.main:load_predict"
        ]
    },
    install_requires=required,
    license="MIT",
)
