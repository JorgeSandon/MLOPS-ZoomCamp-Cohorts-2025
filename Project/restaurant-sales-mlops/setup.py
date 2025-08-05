from setuptools import setup, find_packages

setup(
    name="restaurant-sales-mlops",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "prefect",
        "pandas",
        "scikit-learn",
    ],
)