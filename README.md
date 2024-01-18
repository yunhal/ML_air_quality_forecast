Project Title: Air Quality Forecasting with Machine Learning models


Description:

This repository contains the refactored and cleaned-up versions of machine learning models originally developed for this paper below: 
https://www.frontiersin.org/articles/10.3389/fdata.2023.1124148/full

The original version of the models is available in https://gitlab.com/casus_atm_modeling/ml_multi_site

The primary focus of this repository is to provide well-documented, standardized, and improved versions of these models for better usability and understanding.

Features:

    Refactored Code: The original codebase from the gitlab Repository has been significantly refactored for better readability and maintainability.

    Documentation: Detailed comments and descriptions have been added to improve clarity.

Models

Briefly describe each model included in this repository. For example:

    Model A: random forecast classifiers and multiple linear regression models
    Model B: two-phase random forest regression model


Getting Started

Dependencies: ML_env.yml contains the python libraries need to be installed to run these models. 

Installing: 

------
git clone git@github.com:yunhal/ML_air_quality_forecast.git
cd ML_air_quality_forecast
conda env create -f ML_env.yml
------

Executing Program: 
python prep_input_data.py # to prepare the input data for ML 
python Predict_O3_PM25.py # to compute O3/PM2.5 predictions using the input datasets


Main Authors: Kai Fan and Yunha Lee (PI)
