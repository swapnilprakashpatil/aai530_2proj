import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


class Preprocessor:
    def __init__(self, input_sequence_length, output_sequence_length):
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.data = None
        self.daily_counts = None
        self.station_to_index = {}
        self.scaler = None
        self.station_sequences = {}

        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

        self.x_train_tensor = None
        self.y_train_tensor = None
        self.x_val_tensor = None
        self.y_val_tensor = None
        self.x_test_tensor = None
        self.y_test_tensor = None

    def load_data(self, csv_file_path):
        self.data = pd.read_csv(csv_file_path)
        print(f"Data loaded successfully with shape {self.data.shape}.")

    def convert_dates(self):
        self.data['start_date'] = pd.to_datetime(self.data['started_at'], errors='coerce').dt.date
        self.data['end_date'] = pd.to_datetime(self.data['ended_at'], errors='coerce').dt.date
        print("Converted 'started_at' and 'ended_at' to dates.")

    def compute_daily_counts(self):
        check_outs = self.data.groupby(['start_station_id', 'start_lat', 'start_lng', 'start_date']).size().reset_index(name='check_out_count')
        check_ins = self.data.groupby(['end_station_id', 'end_date']).size().reset_index(name='check_in_count')

        check_outs.rename(columns={'start_station_id': 'station_id', 'start_date': 'date'}, inplace=True)
        check_ins.rename(columns={'end_station_id': 'station_id', 'end_date': 'date'}, inplace=True)

        self.daily_counts = pd.merge(check_outs, check_ins, on=['station_id', 'date'], how='outer')
        self.daily_counts.fillna(0, inplace=True)
        print(f"Computed daily counts with shape {self.daily_counts.shape}.")

    def standardize_coordinates(self):
        station_coords = self.data.groupby('start_station_id').agg({
            'start_lat': 'median',
            'start_lng': 'median'
        }).reset_index()

        self.data = self.data.merge(
            station_coords,
            on='start_station_id',
            how='left',
            suffixes=('_orig', '')
        )
        self.data.drop(['start_lat_orig', 'start_lng_orig'], axis=1, inplace=True)
        print("Coordinates standardized using median values per station.")

    def encode_dates(self):
        self.daily_counts['date'] = pd.to_datetime(self.daily_counts['date'])

        self.daily_counts['day_of_week'] = self.daily_counts['date'].dt.weekday
        self.daily_counts['day_of_month'] = self.daily_counts['date'].dt.day
        self.daily_counts['month'] = self.daily_counts['date'].dt.month

        self.daily_counts['day_of_week_sin'] = np.sin(2 * np.pi * self.daily_counts['day_of_week'] / 7)
        self.daily_counts['day_of_week_cos'] = np.cos(2 * np.pi * self.daily_counts['day_of_week'] / 7)
        self.daily_counts['day_of_month_sin'] = np.sin(2 * np.pi * self.daily_counts['day_of_month'] / 31)
        self.daily_counts['day_of_month_cos'] = np.cos(2 * np.pi * self.daily_counts['day_of_month'] / 31)
        self.daily_counts['month_sin'] = np.sin(2 * np.pi * self.daily_counts['month'] / 12)
        self.daily_counts['month_cos'] = np.cos(2 * np.pi * self.daily_counts['month'] / 12)

        print("Date features encoded cyclically.")

    def split_data(self):
        train_ratio = 0.6
        val_ratio = 0.2

        stations = self.daily_counts['station_id'].unique()

        for station in stations:
            station_df = self.daily_counts[self.daily_counts['station_id'] == station].copy()
            station_df.sort_values('date', inplace=True)

            total_days = len(station_df)
            if total_days > 90:
                train_end = int(train_ratio * total_days)
                val_end = int((train_ratio + val_ratio) * total_days)

                train_data = station_df.iloc[:train_end]
                val_data = station_df.iloc[train_end:val_end]
                test_data = station_df.iloc[val_end:]

                self.station_sequences[station] = {
                    'train': train_data,
                    'val': val_data,
                    'test': test_data
                }

        print("Data split into train, validation, and test sets per station.")

    def scale_features(self):
        features_to_scale = [
            'check_in_count', 'check_out_count',
            'day_of_week_sin', 'day_of_week_cos',
            'day_of_month_sin', 'day_of_month_cos',
            'month_sin', 'month_cos', 'start_lat', 'start_lng',
        ]

        all_train_data = pd.concat([seq['train'] for seq in self.station_sequences.values()])
        self.scaler = MinMaxScaler()
        self.scaler.fit(all_train_data[features_to_scale])

        for station, datasets in self.station_sequences.items():
            for split_name in ['train', 'val', 'test']:
                data = datasets[split_name]
                data_scaled = data.copy()
                data_scaled[features_to_scale] = self.scaler.transform(data[features_to_scale])
                datasets[split_name] = data_scaled

        print("Features scaled using MinMaxScaler.")

    def create_sequences(self):
        features = [
            'check_in_count', 'check_out_count',
            'day_of_week_sin', 'day_of_week_cos',
            'day_of_month_sin', 'day_of_month_cos',
            'month_sin', 'month_cos', 'start_lat', 'start_lng'
        ]
        target = ['check_in_count', 'check_out_count']

        x_train, y_train = [], []
        x_val, y_val = [], []
        x_test, y_test = [], []

        for station, datasets in tqdm(self.station_sequences.items(), desc='Processing station sequences.'):
            for split_name in ['train', 'val', 'test']:
                data = datasets[split_name].reset_index(drop=True)
                input_data = data[features].values
                output_data = data[target].values

                total_length = self.input_sequence_length + self.output_sequence_length
                n_sequences = len(data) - total_length + 1
                n_input_features = len(features)
                n_output_features = len(target)

                if n_sequences < 1:
                    print(f"DataFrame for station {station} is too short to create any complete sequences")
                    continue

                input_stride = input_data.strides[0]
                input_shape = (n_sequences, self.input_sequence_length, n_input_features)
                input_strides = (input_stride, input_stride, input_data.strides[1])
                inputs = np.lib.stride_tricks.as_strided(
                    input_data,
                    shape=input_shape,
                    strides=input_strides,
                    writeable=False
                )

                output_stride = output_data.strides[0]
                output_shape = (n_sequences, self.output_sequence_length, n_output_features)
                output_strides = (output_stride, output_stride, output_data.strides[1])
                outputs = np.lib.stride_tricks.as_strided(
                    output_data[self.input_sequence_length:],
                    shape=output_shape,
                    strides=output_strides,
                    writeable=False
                )

                if split_name == 'train':
                    x_train.extend(inputs)
                    y_train.extend(outputs)
                elif split_name == 'val':
                    x_val.extend(inputs)
                    y_val.extend(outputs)
                elif split_name == 'test':
                    x_test.extend(inputs)
                    y_test.extend(outputs)

        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
        self.x_val = np.array(x_val)
        self.y_val = np.array(y_val)
        self.x_test = np.array(x_test)
        self.y_test = np.array(y_test)

        print("Sequences created for training, validation, and testing.")

    def get_tensors(self):
        self.x_train_tensor = torch.tensor(self.x_train, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
        self.x_val_tensor = torch.tensor(self.x_val, dtype=torch.float32)
        self.y_val_tensor = torch.tensor(self.y_val, dtype=torch.float32)
        self.x_test_tensor = torch.tensor(self.x_test, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(self.y_test, dtype=torch.float32)

        print("Data converted to PyTorch tensors.")
