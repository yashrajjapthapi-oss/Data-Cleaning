# Titanic Dataset Analysis

This notebook performs an exploratory data analysis and preprocessing pipeline on the famous Titanic dataset. The goal is to prepare the data for building a machine learning model to predict passenger survival.

## Table of Contents

- [Project Description](#project-description)
- [Setup Instructions](#setup-instructions)
- [Notebook Sections](#notebook-sections)
- [Data Overview](#data-overview)
- [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
- [Feature Scaling](#feature-scaling)

## Project Description

This project analyzes the Titanic dataset to understand the characteristics of passengers and prepare the data for predictive modeling. It involves handling missing values, identifying and treating outliers, and scaling numerical features.

## Setup Instructions

To run this notebook, you need to:

1.  **Download the dataset:** Obtain the `Titanic-Dataset.csv` file. This dataset is commonly available from sources like Kaggle.
2.  **Upload to Colab:** In the first code cell of the notebook, run the `files.upload()` command and select the `Titanic-Dataset.csv` file from your local machine.
3.  **Install Libraries:** Ensure you have the necessary libraries installed. This notebook primarily uses `pandas`, `numpy`, `matplotlib`, and `sklearn`. These are typically pre-installed in Google Colab.

    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    ```

## Notebook Sections

### Data Overview

-   Loads the `Titanic-Dataset.csv` into a pandas DataFrame.
-   Displays basic information about the DataFrame (`df.info()`) and the first few rows (`df.head()`).
-   Checks for duplicate rows.
-   Separates categorical and numerical columns.
-   Counts unique values in categorical columns.
-   Calculates the percentage of missing values for each column.

### Data Cleaning and Preprocessing

-   Drops irrelevant columns such as 'Name', 'Ticket', and 'Cabin' due to high cardinality or a large number of missing values.
-   Removes rows with missing values in the 'Embarked' column.
-   Fills missing 'Age' values with the mean of the 'Age' column.
-   Visualizes the distribution of 'Age' using a box plot to identify outliers.
-   Applies a 2-standard deviation rule to remove outliers from the 'Age' column, creating `df2` and `df3`.

### Feature Scaling

-   Selects features (`X`) and the target variable (`Y`).
-   Applies `MinMaxScaler` to numerical features in the `X` DataFrame to scale them between 0 and 1.


Feel free to extend this analysis or use the preprocessed data for building your predictive models!
