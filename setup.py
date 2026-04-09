from setuptools import setup, find_packages

setup(
    name="weather-model",
    version="0.1.0",
    description="Python weather forecasting for Polymarket prediction markets",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "scipy>=1.11.0",
        "requests>=2.31.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "statsmodels>=0.14.0",
    ],
)
