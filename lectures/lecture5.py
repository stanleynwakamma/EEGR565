import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def lecture():
    df = pd.read_csv('russian_housing.csv')
    df.head()

    # Size of dataset
    print("Number of rows/columns: ", df.shape)
    print("Data types: ", df.dtypes)

    # Select the numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    numeric_cols = df_numeric.columns.values
    print("Numeric columns: ", numeric_cols)

    # Select the non numeric columns
    df_non_numeric = df.select_dtypes(exclude=[np.number])
    non_numeric_cols = df_non_numeric.columns.values
    print("Non-numeric columns: ", non_numeric_cols)

    matplotlib.rcParams['figure.figsize'] = (12, 8)

    # Technique 1: Missing Data Heatmap
    cols = df.columns[:30] # First 30 columns
    colors = ['#000099', '#ffff00'] # Specify the colors - yellow is missing
    sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colors))
    plt.show()
