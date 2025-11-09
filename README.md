# E-commerce Customer Spending Predictor

A machine learning project that uses Linear Regression to predict yearly customer spending based on various e-commerce metrics.

## Overview

This project analyzes e-commerce customer data to predict yearly spending amounts using features like:
- Average Session Length
- Time spent on App
- Time spent on Website
- Length of Membership

## Features

- Data visualization using Seaborn and Matplotlib
- Linear Regression model implementation using scikit-learn
- Model performance analysis with statsmodels
- Residual analysis and model validation
- Interactive visualization options with both Streamlit and Gradio interfaces

## Requirements

```
matplotlib==3.10.7
seaborn==0.13.2
streamlit==1.51.0
pandas==2.3.3
scikit-learn==1.7.2
statsmodels==0.14.5
numpy==2.3.4
scipy==1.16.3
```

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the main analysis script:
   ```
   python linear_reg.py
   ```

2. For the web interface, run either:
   ```
   python app.py  # For Streamlit interface
   ```
   or
   ```
   python app_gradio.py  # For Gradio interface
   ```

## Model Performance

The model provides:
- High R-squared score on both training and test sets
- Detailed coefficient analysis
- Residual analysis for model validation
- Visual comparison of predictions vs actual values

## Dataset

The project uses the "Ecommerce Customers" dataset, which should be placed in the project root directory.