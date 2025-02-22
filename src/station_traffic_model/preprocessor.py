import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm


class Preprocessor:
    MINIMUM_TRAIN_DAYS = 60

    columns_to_load = [
        'started_at', 'start_station_id', 'start_lat', 'start_lng',
    ]
    date_columns = ['started_at']

    train_dataset = None
    val_dataset = None
    test_dataset = None

    def preprocess_data(self, data_path: str, train_proportion: float = 0.6,
                        validation_proportion: float = 0.2,
                        input_sequence_length: int = 30, output_sequence_length: int = 1):
        # Load Data
        data = pd.read_csv(data_path, usecols=self.columns_to_load, parse_dates=self.date_columns)
        data['start_station_id'] = data['start_station_id'].astype('string')
        data['date'] = pd.to_datetime(data['started_at'], errors='coerce').dt.date
        data.rename(columns={'start_station_id': 'station_id', 'start_lat': 'latitude', 'start_lng': 'longitude'},
                    inplace=True)
        print(f'Empty values: {data.isna().sum()}')
        data.dropna(inplace=True)

        # Standardize coordinates
        station_coords = data.groupby('station_id').agg({'latitude': 'median', 'longitude': 'median'}).reset_index()
        data = data.merge(station_coords, on='station_id', how='left', suffixes=('_orig', ''))
        data.drop(['latitude_orig', 'longitude_orig'], axis=1, inplace=True)

        # Group to get counts
        data = data.groupby(['station_id', 'latitude', 'longitude', 'date']).size().reset_index(name='count')

        # Encoding
        label_encoder = LabelEncoder()
        data['station_id'] = label_encoder.fit_transform(data['station_id'])

        data['date'] = pd.to_datetime(data['date'])
        data['day_of_week'] = data['date'].dt.weekday
        data['day_of_month'] = data['date'].dt.day
        data['month'] = data['date'].dt.month
        data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        data['day_of_month_sin'] = np.sin(2 * np.pi * data['day_of_month'] / 31)
        data['day_of_month_cos'] = np.cos(2 * np.pi * data['day_of_month'] / 31)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        data.drop(['day_of_week', 'day_of_month', 'month'], axis=1, inplace=True)

        # Remove stations outside training
        train_rows = int(len(data) * train_proportion)
        train_subset = data.iloc[:train_rows]
        train_subset = train_subset.groupby('station_id').size().reset_index(name='count')
        station_ids_to_include = train_subset[train_subset['count'] > 60]['station_id'].tolist()
        data = data[data['station_id'].isin(station_ids_to_include)]

        # Split data
        train, x_temp = train_test_split(data, train_size=train_proportion, shuffle=False)
        val, test = train_test_split(x_temp, train_size=validation_proportion / (1 - train_proportion), shuffle=False)
        scaler = StandardScaler()
        scaler.fit(train.drop('date', axis=1))

        # Create sequences
        x_train, y_train = self.create_sequences(train, scaler, seq_length=input_sequence_length,
                                                 target_length=output_sequence_length)
        x_val, y_val = self.create_sequences(val, scaler, seq_length=input_sequence_length,
                                             target_length=output_sequence_length)
        x_test, y_test = self.create_sequences(test, scaler, seq_length=input_sequence_length,
                                               target_length=output_sequence_length)

        # Convert to tensors
        self.train_dataset = data_utils.TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
        self.val_dataset = data_utils.TensorDataset(torch.FloatTensor(x_val), torch.FloatTensor(y_val))
        self.test_dataset = data_utils.TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))

    @staticmethod
    def create_sequences(data: pd.DataFrame, scaler: StandardScaler, seq_length: int, target_length: int):
        x, y = [], []
        station_ids = data['station_id'].unique()
        for station_id in tqdm(station_ids):
            x_data = data[data['station_id'] == station_id].sort_values('date', ascending=True)
            x_data.drop(['date'], axis=1, inplace=True)
            y_data = x_data[['count']]

            x_values = scaler.transform(x_data)
            y_values = y_data.values

            for i in range(len(x_data) - seq_length - target_length + 1):
                x.append(x_values[i:(i + seq_length)])
                y.append(y_values[(i + seq_length):(i + seq_length + target_length)])

        return np.array(x), np.array(y)

    def get_loaders(self, batch_size: int):
        train_loader = data_utils.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = data_utils.DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = data_utils.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
