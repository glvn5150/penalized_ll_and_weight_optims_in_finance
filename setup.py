from setuptools import setup, find_packages

setup(
    name='exercises',
    version="0,1",
    packages = find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "yfinance",
        "matplotlib",

        "scikit-learn",
        "seaborn",
        "plotly",
        "bs4",
        "statsmodels",
    ],
)