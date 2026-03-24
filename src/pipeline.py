import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import yaml


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def load_and_clean(data_path, config):
    df = pd.read_csv(data_path, parse_dates=['Date'])
    df = df.sort_values(['Store ID', 'Product ID', 'Date']).reset_index(drop=True)
    frac = config['sampling']['frac']
    seed = config['sampling']['random_state']
    if frac < 1.0:
        combos = df[['Store ID', 'Product ID']].drop_duplicates().sample(frac=frac, random_state=seed)
        df = df.merge(combos, on=['Store ID', 'Product ID'])
        df = df.sort_values(['Store ID', 'Product ID', 'Date']).reset_index(drop=True)
    cat_cols = ['Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']
    df[cat_cols] = df[cat_cols].astype('category')
    df['Holiday/Promotion'] = df['Holiday/Promotion'].astype(bool)
    df['Demand_Forecast_Clean'] = df['Demand Forecast'].clip(lower=1)
    return df


def reconstruct_inventory(group):
    group = group.sort_values('Date').copy()
    sold = group['Units Sold'].values
    ordered = group['Units Ordered'].values
    n = len(group)
    inv = np.empty(n, dtype=np.float64)
    inv[0] = group['Inventory Level'].iloc[0]
    for i in range(1, n):
        inv[i] = max(inv[i-1] - sold[i-1] + ordered[i-1], 0.0)
    group['Inventory_Reconstructed'] = inv
    return group


def build_features(df):
    reconstructed = []
    for (_, _), group in df.groupby(['Store ID', 'Product ID'], sort=False):
        reconstructed.append(reconstruct_inventory(group))
    df = pd.concat(reconstructed).sort_values(['Store ID', 'Product ID', 'Date']).reset_index(drop=True)
    df['Inventory_Lag1'] = df.groupby(['Store ID', 'Product ID'])['Inventory_Reconstructed'].shift(1)
    df['Units_Sold_Lag1'] = df.groupby(['Store ID', 'Product ID'])['Units Sold'].shift(1)
    df['Rolling7_Inventory'] = df.groupby(['Store ID', 'Product ID'])['Inventory_Reconstructed'].transform(lambda x: x.shift(1).rolling(7).mean())
    df['Inventory_Change'] = df['Inventory_Reconstructed'] - df['Inventory_Lag1']
    df['Inventory_Change_Pct'] = (df['Inventory_Change'] / df['Inventory_Lag1'].replace(0, np.nan)).fillna(0)
    df['Days_of_Stock'] = (df['Inventory_Reconstructed'] / df['Units Sold'].replace(0, np.nan)).fillna(df['Inventory_Reconstructed'] / df['Demand_Forecast_Clean'])
    df['Days_of_Stock'] = df['Days_of_Stock'].replace([np.inf, -np.inf], np.nan)
    df['Inventory_vs_Rolling7'] = df['Inventory_Reconstructed'] - df['Rolling7_Inventory']
    df['Sales_Velocity'] = (df['Units Sold'] / df['Inventory_Reconstructed'].replace(0, np.nan)).fillna(0).replace([np.inf, -np.inf], np.nan)
    df['Coverage_Ratio'] = df['Inventory_Reconstructed'] / df['Demand_Forecast_Clean']
    df['Forecast_Error'] = df['Units Sold'] - df['Demand Forecast']
    df['Order_to_Inventory'] = (df['Units Ordered'] / df['Inventory_Reconstructed'].replace(0, np.nan)).fillna(0).replace([np.inf, -np.inf], 0)
    df = df.dropna(subset=['Inventory_Lag1', 'Rolling7_Inventory']).reset_index(drop=True)
    return df


def label_risk(df, config):
    theta_low = config['thresholds']['theta_low']
    theta_high = config['thresholds']['theta_high']
    sales_vel = config['thresholds']['sales_vel']
    df['Risk_Label_Current'] = 'Safe Zone'
    df.loc[df['Inventory Level'] < df['Demand_Forecast_Clean'] * theta_low, 'Risk_Label_Current'] = 'Stockout Risk'
    df.loc[(df['Inventory Level'] > df['Demand_Forecast_Clean'] * theta_high) & (df['Units Sold'] < df['Demand_Forecast_Clean'] * sales_vel), 'Risk_Label_Current'] = 'Overstock Risk'
    df['Risk_Label'] = df.groupby(['Store ID', 'Product ID'])['Risk_Label_Current'].shift(-1)
    df = df.dropna(subset=['Risk_Label']).reset_index(drop=True)
    return df


def split_data(df, config):
    cutoff_val = config['splits']['cutoff_val']
    cutoff_test = config['splits']['cutoff_test']
    train = df[df['Date'] < pd.Timestamp(cutoff_val)].copy()
    val = df[(df['Date'] >= pd.Timestamp(cutoff_val)) & (df['Date'] < pd.Timestamp(cutoff_test))].copy()
    test = df[df['Date'] >= pd.Timestamp(cutoff_test)].copy()
    for col in ['Days_of_Stock', 'Sales_Velocity']:
        train_median = train[col].median()
        train[col] = train[col].fillna(train_median)
        val[col] = val[col].fillna(train_median)
        test[col] = test[col].fillna(train_median)
    categorical_cols = ['Category', 'Region', 'Weather Condition', 'Seasonality']
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(train[col])
        for split in [train, val, test]:
            split[col + '_enc'] = le.transform(split[col])
        encoders[col] = le
    return train, val, test, encoders
