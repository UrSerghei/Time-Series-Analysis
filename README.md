# Time-Series Analysis Project

## Overview
This project focuses on time-series analysis using machine learning techniques to forecast adjusted closing prices of a financial dataset. The primary files in this project include `forec_us_reg.ipynb`, `forec_us_clas.ipynb`, and `VOO-1.csv`, along with a GUI implementation using PyQt.

## Table of Contents
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Exploring Dataset and Visualization](#exploring-dataset-and-visualization)
  - [Description](#description)
  - [Exploring Dataset](#exploring-dataset)
  - [Checking and Plotting Null Values](#checking-and-plotting-null-values)
  - [Checking Data Set Information](#checking-data-set-information)
  - [Plotting Correlation](#plotting-correlation)
  - [Converting into Meaningful Strings](#converting-into-meaningful-strings)
  - [Distributions](#distributions)
    - [Distribution of Year](#distribution-of-year)
    - [Distribution of Day](#distribution-of-day)
    - [Distribution of Month](#distribution-of-month)
    - [Distribution of Quarter](#distribution-of-quarter)
  - [Scatter Distributions](#scatter-distributions)
    - [Open vs Adjusted Close vs Year](#scatter-distribution-of-open-vs-adj-close-vs-year)
    - [Low vs High vs Quarter](#scatter-distribution-of-low-vs-high-vs-quarter)
    - [Adjusted Close vs Volume vs Month](#scatter-distribution-of-adj-close-vs-volume-vs-month)
    - [High vs Volume vs Day](#scatter-distribution-of-high-vs-volume-vs-day)
  - [Volume Distributions](#volume-distributions)
    - [Distribution of Volume by Year](#distribution-of-volume-by-year)
    - [Distribution of Volume by Day](#distribution-of-volume-by-day)
    - [Distribution of Volume by Month](#distribution-of-volume-by-month)
    - [Distribution of Volume by Quarter](#distribution-of-volume-by-quarter)
- [Forecasting Adjusted Closing Using Machine Learning](#forecasting-adjusted-closing-using-machine-learning)
  - [Computing Technical Indicators](#computing-technical-indicators)
  - [Regression Models](#regression-models)
- [Predicting Stock Daily Return Using Machine Learning](#predicting-stock-daily-return-using-machine-learning)
  - [Normalizing Data](#normalizing-data)
  - [Modeling Techniques](#modeling-techniques)
- [Graphical User Interface Implementation](#graphical-user-interface-implementation)
  - [Files](#files)
- [Conclusion](#conclusion)
- [License](#license)

## Project Structure
- `forec_us_reg.ipynb`: Jupyter notebook containing time-series analysis and forecasting procedures.
- `forec_us_clas.ipynb`: Jupyter notebook for predicting stock daily returns using machine learning.
- `VOO-1.csv`: Historical dataset of S&P 500 stock prices.
- `gui_stock.py`: Python code for the GUI implementation.
- `gui_yahoo.ui`: GUI design file.
- `plot_class.py`: Additional class for the GUI.

## Getting Started
To run this project, you will need to have Jupyter Notebook installed along with the required libraries. You can install the necessary libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn PyQt5
```

## Exploring Dataset and Visualization

### Description
This section provides an overview of the dataset, including its structure and key features. The analysis focuses on understanding the trends and patterns present in the time-series data.

### Exploring Dataset
- Load the dataset and display the first few rows to understand its structure.
- Summary statistics of the dataset will be provided to identify key features.

### Checking and Plotting Null Values
- Analyze and visualize the presence of null values in the dataset.
- Plot graphs to illustrate the distribution of missing data and identify patterns.

### Checking Data Set Information
- Display information about the dataset, including data types, non-null counts, and memory usage.

### Plotting Correlation
- Create correlation heatmaps to visualize the relationships between different features.

### Converting into Meaningful Strings
- Convert numerical date features into meaningful string representations for better interpretability.

### Distributions
#### Distribution of Year
- Plot the distribution of years in the dataset.

#### Distribution of Day
- Plot the distribution of days in the dataset.

#### Distribution of Month
- Plot the distribution of months in the dataset.

#### Distribution of Quarter
- Plot the distribution of quarters in the dataset.

### Scatter Distributions
#### Scatter Distribution of Open vs Adjusted Close vs Year
- Visualize the relationship between opening and adjusted closing prices over the years.

#### Scatter Distribution of Low vs High vs Quarter
- Analyze the relationship between low and high prices across different quarters.

#### Scatter Distribution of Adjusted Close vs Volume vs Month
- Explore the relationship between adjusted closing prices and trading volume by month.

#### Scatter Distribution of High vs Volume vs Day
- Investigate the relationship between high prices and trading volume by day.

### Volume Distributions
#### Distribution of Volume by Year
- Plot the distribution of trading volume by year.

#### Distribution of Volume by Day
- Plot the distribution of trading volume by day.

#### Distribution of Volume by Month
- Plot the distribution of trading volume by month.

#### Distribution of Volume by Quarter
- Plot the distribution of trading volume by quarter.

## Forecasting Adjusted Closing Using Machine Learning
This section focuses on building and training machine learning models to forecast adjusted closing prices based on the explored features.

### Computing Technical Indicators
- Compute various technical indicators, including MACD, RSI, SMA, and Bollinger Bands, to enhance forecasting accuracy.

### Scatter Distribution of Adjusted Close versus Daily Returns versus Year
- Visualize the relationship between adjusted closing prices and daily returns across different years.

### Scatter Distribution of Volume versus Daily Returns versus Year
- Analyze the relationship between trading volume and daily returns over the years.

### Regression Models
- **Preprocessing Data**: Prepare the data for modeling by cleaning and transforming the dataset.
- **Performing Regression**: Implement various regression techniques, including:
  - Linear Regression
  - Random Forest Regression
  - Decision Tree Regression
  - K-Nearest Neighbors Regression
  - AdaBoost Regression
  - Gradient Boosting Regression
  - Extreme Gradient Boosting Regression
  - Light Gradient Boosting Regression
  - CatBoost Regression
  - Support Vector Regression
  - Lasso Regression
  - Ridge Regression
  - Multi-Layer Perceptron Regression

## Predicting Stock Daily Return Using Machine Learning
This section focuses on predicting daily stock returns using various classification models.

### Normalizing Data
- Normalize the dataset to ensure all features contribute equally to the model.

### Learning Curve
- Generate learning curves to assess the performance of the model as training size increases.

### Real Values Versus Predicted Values Diagram and Confusion Matrix
- Visualize the comparison between actual and predicted values and create a confusion matrix for model evaluation.

### Decision Boundaries and ROC
- Plot decision boundaries and ROC curves to analyze model performance.

### Modeling Techniques
Implement various classification techniques, including:
- Support Vector Classifier
- Logistic Regression Classifier
- K-Nearest Neighbors Classifier
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- AdaBoost Classifier
- Extreme Gradient Boosting Classifier
- Multi-Layer Perceptron Classifier
- Light Gradient Boosting Classifier
- Gaussian Mixture Model Classifier
- Extra Trees Classifier
- LSTM and GRU

## Graphical User Interface Implementation
This section provides an overview of the GUI implementation using PyQt.

### Files
- `gui_stock.py`: Contains the main Python code for the GUI application.
- `gui_yahoo.ui`: The UI design file created using Qt Designer.
- `plot_class.py`: Contains additional classes and functions to support the GUI.

### GUI Features
- Designing GUI
- Preprocessing data and populating tables
- Plotting various distributions and features
- Preparing data for forecasting
- Implementing regression models for predictions

## Conclusion
This project demonstrates the application of machine learning in forecasting time-series data, with a specific focus on financial metrics. Through exploration, visualization, and modeling, we aim to derive meaningful insights and predictions from the dataset.
