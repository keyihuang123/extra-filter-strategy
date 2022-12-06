# extra-filter-strategy
Trade Filtering Method for Trend Following Strategy  based on LSTM-extracted Feature and Machine Learning

### Background
The project is the code for paper 'Trade Filtering Method for Trend Following Strategy based on LSTM-extracted Feature and Machine Learning'.
The idea of the 'LSTM feature extraction and machine learning classification' mentioned in the paper will be implemented using Python and the strategy backtesting using other platforms.

### File Description

##### data

* File name: 'RB00_5M_sample.csv'

* File description: This is a small sample of data for the rebar futures contract, with a time period of 30th Dec 2018 to 28th Jun 2022.
 
##### src -- feature

* The paper mentions that we use three feature groups, and the following file paths represent each of the three different feature groups:

  1. base group (raw data): \data\ **'RB00_5M_sample.csv'**

  2. TsFeature group (Time series feature): \src\feature\ **kats**

  3. TA-Lib group (Technical indicators feature): \src\feature\ **ta_lib**

* We use the LSTM feature extraction method with the file path: \src\feature\ **lstm**

##### src -- models

* We have used a machine learning model for the classification task, in the file：\src\models\ **classification.py**

* Since we plan to explore the regression model in the future, but it is not currently mentioned in this paper, this \src\models\regression.py can be ignored

##### src -- pipline

* Create category labels: \src\pipeline\ **label.py**

* Merge labels and pre-processed features: \src\pipeline\ **merge.py**

* Preprocessing of features: \src\pipeline\ **preprocessing.py**

##### src

* **main.py**: run the entire project and define the window length for the sliding window segmentation dataset

* **train.py**: invoke data integration, feature engineering, ML classification models and data pre-processing modules

* **utils.py**: collation of forecast results and forecast evaluation results

##### config

* **config.yaml**: configuration file with the possibility to make changes to the adjustable parameters
















