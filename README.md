# ITDS Final Project

**By**: Arron Pirku  
**Neptun Code**: MT94J3  

---

## Project Overview

This project implements and evaluates various regression techniques on both synthetic and real-world datasets. It is divided into three main sections:

### Section 2.1: Univariate Regression on Analytical Functions
- Function used: `f1(x) = x·sin(x) + 2x`
- Models tested: Linear Regression, Ridge Regression, SVR, Random Forest, MLP
- Advanced feature engineering using polynomial and trigonometric terms
- Additional test for model robustness by injecting Gaussian noise into the data

### Section 2.2: Multivariate Regression on Synthetic Data
- Dataset generated using `make_regression` from scikit-learn with 10 features and partial informativeness
- Models compared for multivariate regression performance under noise

### Section 2.3: Time Series Forecasting Using WWII Weather Data
- Real dataset from NOAA (sensor ID 22508, Honolulu)
- Preprocessed for date alignment and missing value handling
- Created rolling window datasets to predict next-day temperature
- Compared window sizes of 3, 5, 7, 14, and 30 using Random Forest Regressor

---

## Function and Logic Overview

### `f1(x)`
Defines the target univariate function used for synthetic regression:  
  **f1(x) = x·sin(x) + 2x**

### `enhanced_features(x)`
Constructs a richer feature set using polynomial and trigonometric transformations:  
  Includes x, x², sin(x), cos(x), and x·sin(x)

### `inject_noise(y)`
Adds Gaussian noise to the target vector `y` using a mean of 0 and standard deviation of 50  
  Used to test model robustness to noisy data

### Model Definitions and Evaluation
A set of five models (Linear, Ridge, SVR, Random Forest, MLP) are defined and trained on the data.  
Each model is evaluated using MSE and R² metrics.

### Multivariate Data Generation
Synthetic dataset is generated with 2000 samples, 10 features (5 informative), and evaluated on the same regression models.

### WWII Temperature Forecasting Logic
- The dataset is downloaded, filtered for Honolulu, and the mean temperature is extracted
- A rolling window approach is used to create training features for next-day prediction
- Models are trained and evaluated year-wise, using 1940–1944 for training and 1945 for testing
- A final comparison loop evaluates model performance across different window sizes (3 to 30 days)

---

This notebook was developed entirely in Google Colab using `numpy`, `pandas`, `matplotlib`, and `scikit-learn`.
