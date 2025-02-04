import os

import pandas as pd
import torch


class Preprocessor:
    @staticmethod
    def get_preprocessed_data(data_path: str):
        cache_file = data_path.replace('.csv', '_preprocessed.csv')
        if os.path.exists(cache_file):
            print(f"Loading preprocessed data from {cache_file}")
            return pd.read_csv(cache_file, parse_dates=['start_date', 'end_date'])

        print(f"Preprocessing data from {data_path}")
        df = pd.read_csv(data_path)

        df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
        df['ended_at'] = pd.to_datetime(df['ended_at'], errors='coerce')

        df['start_date'] = df['started_at'].dt.date
        df['end_date'] = df['ended_at'].dt.date

        check_outs = df.groupby(['start_station_id', 'start_date']).size().reset_index(name='check_outs')
        check_ins = df.groupby(['end_station_id', 'end_date']).size().reset_index(name='check_ins')
        data = pd.merge(check_ins, check_outs, left_on=['end_station_id', 'end_date'],
                        right_on=['start_station_id', 'start_date'], how='outer')

        data['check_ins'] = data['check_ins'].fillna(0)
        data['check_outs'] = data['check_outs'].fillna(0)

        data.to_csv(cache_file, index=False)
        print(f"Preprocessed data saved to {cache_file}")

        return data

    @staticmethod
    def get_sequences(data, input_window, output_window):
        sequences = []
        for i in range(len(data) - input_window - output_window + 1):
            seq_input = data[i:(i + input_window)]
            seq_output = data[(i + input_window):(i + input_window + output_window)]
            sequences.append((seq_input, seq_output))
        return sequences

    @staticmethod
    def get_station_data(data, station_id):
        station_data = data[(data['end_station_id'] == station_id) | (data['start_station_id'] == station_id)].copy()

        station_data['date'] = station_data['end_date'].combine_first(station_data['start_date'])
        station_data = station_data[['date', 'check_ins', 'check_outs']]

        station_data.set_index('date', inplace=True)
        station_data = station_data.resample('D').sum()
        station_data.reset_index(inplace=True)
        return station_data

    @staticmethod
    def convert_sequences_to_tensors(sequences):
        x = [torch.tensor(s[0][['check_ins', 'check_outs']].values, dtype=torch.float32) for s in sequences]
        y = [torch.tensor(s[1][['check_ins', 'check_outs']].values, dtype=torch.float32) for s in sequences]
        return x, y
