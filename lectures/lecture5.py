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
    # plt.show()

    # Technique 2: Missing Data percentage list
    missing_data = [(col, round(np.mean(df[col].isnull())*100)) for col in df.columns]
    missing_data = sorted(missing_data, key=lambda x: x[1], reverse=True)
    for data in missing_data:
        if (data[1] == 0):
            break
        print('{} - {}%'.format(data[0], data[1]))

    # Technique 3:
    # first create missing indicator for features with missing data
    for col in df.columns:
        missing = df[col].isnull()
        num_missing = np.sum(missing)

        if num_missing > 0:
            df['{}_ismissing'.format(col)] = missing

    # then bassed on the indicator, plot the histogram of missing values
    ismissing_cols = [col for col in df.columns if 'ismissing' in col]
    df['num_missing'] = df[ismissing_cols].sum(axis=1)
    df['num_missing'].value_counts().reset_index().sort_values(by='index').plot.bar(x='index', y='num_missing')
    plt.show()

    # drop rows with a lot of missing values
    ind_missing = df[df['num_missing'] > 35].index
    df_less_missing_rows = df.drop(ind_missing, axis=0)
