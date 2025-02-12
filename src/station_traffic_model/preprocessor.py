import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


class Preprocessor:
    @staticmethod
    def get_preprocessed_data(data_path: str):
        cache_file = data_path.replace('.csv', '_preprocessed.csv')
        if os.path.exists(cache_file):
            print(f"Loading preprocessed data from {cache_file}")
            return pd.read_csv(cache_file)
        else:
            print(f"Preprocessing data from {data_path}")
            data = pd.read_csv(data_path)

            data['started_at'] = pd.to_datetime(data['started_at'], errors='coerce')
            data['ended_at'] = pd.to_datetime(data['ended_at'], errors='coerce')

            data['start_date'] = data['started_at'].dt.date
            data['start_dayofweek'] = data['started_at'].dt.dayofweek
            data['start_month'] = data['started_at'].dt.month

            data['end_date'] = data['ended_at'].dt.date
            data['end_dayofweek'] = data['ended_at'].dt.dayofweek
            data['end_month'] = data['ended_at'].dt.month

            check_outs = data.groupby(['start_station_id', 'start_date']).size().reset_index(name='check_out_count')
            check_ins = data.groupby(['end_station_id', 'end_date']).size().reset_index(name='check_in_count')
            check_outs.rename(columns={'start_station_id': 'station_id', 'start_date': 'date'}, inplace=True)
            check_ins.rename(columns={'end_station_id': 'station_id', 'end_date': 'date'}, inplace=True)

            daily_counts = pd.merge(check_outs, check_ins, on=['station_id', 'date'], how='outer')
            daily_counts['check_out_count'].fillna(0, inplace=True)
            daily_counts['check_in_count'].fillna(0, inplace=True)

            date_range = pd.date_range(start=data['started_at'].min().date(), end=data['ended_at'].max().date())
            stations = daily_counts['station_id'].unique()
            full_index = pd.MultiIndex.from_product([stations, date_range], names=['station_id', 'date'])
            full_daily_counts = daily_counts.set_index(['station_id', 'date']).reindex(full_index,
                                                                                       fill_value=0).reset_index()

            full_daily_counts['dayofweek'] = full_daily_counts['date'].apply(lambda x: x.weekday())
            full_daily_counts['month'] = full_daily_counts['date'].apply(lambda x: x.month)
            full_daily_counts['is_weekend'] = full_daily_counts['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

            day_of_week_encoded = pd.get_dummies(full_daily_counts['dayofweek'], prefix='day')
            full_daily_counts = pd.concat([full_daily_counts, day_of_week_encoded], axis=1)
            full_daily_counts.drop('dayofweek', axis=1, inplace=True)

            month_encoded = pd.get_dummies(full_daily_counts['month'], prefix='month')
            full_daily_counts = pd.concat([full_daily_counts, month_encoded], axis=1)
            full_daily_counts.drop('month', axis=1, inplace=True)

            scaler_check_out = MinMaxScaler()
            scaler_check_in = MinMaxScaler()
            full_daily_counts['check_out_count'] = scaler_check_out.fit_transform(
                full_daily_counts[['check_out_count']])
            full_daily_counts['check_in_count'] = scaler_check_in.fit_transform(full_daily_counts[['check_in_count']])

            full_daily_counts['day_of_month'] = full_daily_counts['date'].apply(lambda x: x.day)
            scaler_dayofmonth = MinMaxScaler()
            full_daily_counts['day_of_month'] = scaler_dayofmonth.fit_transform(full_daily_counts[['day_of_month']])

            full_daily_counts['year'] = full_daily_counts['date'].apply(lambda x: x.year)
            scaler_year = MinMaxScaler()
            full_daily_counts['year'] = scaler_year.fit_transform(full_daily_counts[['year']])

            joblib.dump(scaler_check_out, 'saved_models/scalers/scaler_check_out.save')
            joblib.dump(scaler_check_in, 'saved_models/scalers/scaler_check_in.save')
            joblib.dump(scaler_dayofmonth, 'saved_models/scalers/scaler_dayofmonth.save')
            joblib.dump(scaler_year, 'saved_models/scalers/scaler_year.save')
            full_daily_counts.to_csv(cache_file, index=False)
            print(f"Preprocessed data saved to {cache_file}")

            return full_daily_counts

    @staticmethod
    def create_sequences(data, seq_length):
        xs = []
        ys = []
        for station_id in data['station_id'].unique():
            station_data = data[data['station_id'] == station_id].sort_values('date')
            features = station_data.drop(['station_id', 'date'], axis=1).values.astype(np.float32)
            for i in range(len(features) - seq_length):
                x = features[i:(i + seq_length)]
                y = features[i + seq_length][:2]
                xs.append(x)
                ys.append(y)
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    @staticmethod
    def split_data(x, y, train_proportion, val_proportion):
        train_size = int(len(x) * train_proportion)
        val_size = int(len(x) * val_proportion)

        x_train = torch.from_numpy(x[:train_size]).float()
        y_train = torch.from_numpy(y[:train_size]).float()
        x_val = torch.from_numpy(x[train_size: train_size + val_size]).float()
        y_val = torch.from_numpy(y[train_size: train_size + val_size]).float()
        x_test = torch.from_numpy(x[train_size + val_size:]).float()
        y_test = torch.from_numpy(y[train_size + val_size:]).float()

        return x_train, y_train, x_val, y_val, x_test, y_test
