import pandas as pd
import numpy as np


def load_data(path):
    df = pd.read_csv(path)
    return df


def create_lag_features(df):
    df['PRCP_Lag1'] = df['PRCP'].shift(1)
    df['PRCP_Lag2'] = df['PRCP'].shift(2)

    df['TAVG_Lag1'] = df['TAVG'].shift(1)
    df['TAVG_Lag2'] = df['TAVG'].shift(2)

    return df


def create_rolling_features(df):
    df['PRCP_3day_avg'] = df['PRCP'].rolling(window=3).mean()
    df['PRCP_7day_avg'] = df['PRCP'].rolling(window=7).mean()
    df['PRCP_14day_avg'] = df['PRCP'].rolling(window=14).mean()

    return df


def create_cyclical_features(df):
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
    df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)

    return df


def create_target(df):
    df['RainTomorrow'] = (df['PRCP'].shift(-1) > 0).astype(int)
    return df


def clean_data(df):
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df


def split_data(df, split_year=2019):
    train = df[df['Year'] < split_year]
    test = df[df['Year'] == split_year]

    X_train = train.drop(columns=['RainTomorrow'])
    y_train = train['RainTomorrow']

    X_test = test.drop(columns=['RainTomorrow'])
    y_test = test['RainTomorrow']

    return X_train, X_test, y_train, y_test


def full_preprocessing_pipeline(path):
    df = load_data(path)
    df = create_target(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_cyclical_features(df)
    df = clean_data(df)

    X_train, X_test, y_train, y_test = split_data(df)

    return X_train, X_test, y_train, y_test
