from setuptools import find_packages, setup

setup(
    name='SalesMasterML',
    packages=find_packages(),
    version='0.1.0',
    description='Machine Learning models for sales prediction and forecasting in retail.',
    author='Vishal Raj',
    license='MIT',
    install_requires=[
        'pandas==1.4.2',
        'scikit-learn==1.0.2',  # Note that the package name is scikit-learn, not sklearn
        'matplotlib==3.7.2',
        'hyperopt==0.2.7',
        'joblib==1.2.0',
        'prophet==1.1.4',
        'ipywidgets==8.0.4',
        'numpy==1.22',
        'plotly==5.9.0',
        'xgboost==1.7.3',
        'fastapi==0.99.1',
        'pydantic==1.10.12',
        'uvicorn==0.23.2',
    ]
)