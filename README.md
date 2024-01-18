P# Air Quality Forecasting with Machine Learning Models

## Description

This repository contains the refactored and cleaned-up versions of machine learning models originally developed for the paper: [Air Quality Forecasting with Machine Learning](https://www.frontiersin.org/articles/10.3389/fdata.2023.1124148/full). The original version of the models is available at [GitLab Repository](https://gitlab.com/casus_atm_modeling/ml_multi_site).

The primary focus of this repository is to provide well-documented, standardized, and improved versions of these models for better usability and understanding.

## Features

- **Refactored Code:** The original codebase from the GitLab repository has been significantly refactored for better readability and maintainability.
- **Documentation:** Detailed comments and descriptions have been added to improve clarity.

## Models

This repository includes the following models:

- **Model A:** Random forecast classifiers and multiple linear regression models.
- **Model B:** Two-phase random forest regression model.

## Getting Started

### Dependencies

`ML_env.yml` contains the Python libraries needed to run these models.

### Installing

To set up the project, run the following commands:

```bash
git clone git@github.com:yunhal/ML_air_quality_forecast.git
cd ML_air_quality_forecast
conda env create -f ML_env.yml
```

### Executing Program: 
* To prepare the input data for machine learning:
```
python prep_input_data.py
```

*To compute O3/PM2.5 predictions using the input datasets:
```
python Predict_O3_PM25.py
```


### Main Authors: 
* Kai Fan
* Yunha Lee (PI)
