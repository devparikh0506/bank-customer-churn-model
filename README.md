# Bank Customer Churn Prediction Model

This project implements a machine learning model to predict customer churn for a bank using various customer attributes and behaviors.

## Project Overview

The goal of this project is to develop a predictive model that can identify customers who are likely to leave the bank (churn). This can help the bank take proactive measures to retain at-risk customers.

## Dataset

The dataset contains information about bank customers, including:

- Customer ID
- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary
- Churn (target variable)

## Model Architecture

The project uses a neural network implemented in PyTorch. The model architecture includes:

- Input layer
- Multiple hidden layers with ReLU activation
- Output layer with sigmoid activation for binary classification

## Project Structure

The Jupyter notebook contains the following main sections:

1. **Data Loading and Preprocessing**: Loading the dataset and preparing it for model training.
2. **Model Definition**: Creating a custom neural network for churn prediction.
3. **Training**: Implementing the training loop with optimization.
4. **Evaluation**: Assessing the model's performance on a test set.

## Requirements

- Python 3.x
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib (for visualizations)

## Usage

1. Ensure all required libraries are installed.
2. Open and run the Jupyter notebook `model.ipynb`.
3. Follow the notebook cells to train and evaluate the churn prediction model.

## Performance

The model's performance metrics (such as accuracy, precision, recall, and F1-score) are calculated and displayed in the notebook.

## Future Improvements

- Experiment with different model architectures or ensemble methods.
- Perform feature engineering to potentially improve model performance.
- Implement cross-validation for more robust evaluation.
- Explore interpretability techniques to understand key factors influencing churn.
