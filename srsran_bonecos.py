import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json as jason
from sklearn.preprocessing import minmax_scale, MinMaxScaler
"""from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split"""

def kpm_load():
    return pd.read_pickle('./pickles/srsran_kpms/df_kpms.pkl')

def iperf_load():
    return pd.read_pickle('./pickles/srsran_kpms/df_iperf.pkl')

def latency_load():
    return pd.read_pickle('./pickles/srsran_kpms/df_latency.pkl')

def load_timestamps():
    with open('./helpers/timestamps.json', 'r') as file:
        timestamps = jason.load(file)
    return timestamps['srs_tests']

def filter_dataframe_by_timestamps(df, start_timestamp, end_timestamp):
    return df[(df['_time'] >= start_timestamp) & (df['_time'] <= end_timestamp)]

def filter_dataframe_by_test(df, test_conditions, test_nr, timestamps):
    start_timestamp = timestamps[test_conditions][test_nr]['start']
    end_timestamp = timestamps[test_conditions][test_nr]['end']
    return filter_dataframe_by_timestamps(df, start_timestamp, end_timestamp)

def kpm_plot(df_kpm, df_iperf, df_latency):
    df_kpm = df_kpm[df_kpm['DRB.RlcSduTransmittedVolumeUL'] > 5]

    print(df_kpm.head())

    df_kpm_normalized = minmax_scale(df_kpm['DRB.RlcSduTransmittedVolumeUL'])
    df_iperf_normalized = minmax_scale(df_iperf['path_loss'])
    df_latency_normalized = minmax_scale(df_latency['time_latency'])

    plt.plot(df_kpm['_time'], df_kpm_normalized, '.', label='DRB.RlcSduTransmittedVolumeUL')
    plt.plot(df_iperf['_time'], df_iperf_normalized, '.', label='Path Loss')
    plt.plot(df_latency['_time'], df_latency_normalized, '.', label='Latency')

    plt.legend()
    plt.show()

def prepare_lstm_data(df_kpm, df_iperf, df_latency, time_steps):
    scaler = MinMaxScaler()
    df_kpm_normalized = scaler.fit_transform(df_kpm.values.reshape(-1, 1))
    df_iperf_normalized = scaler.fit_transform(df_iperf.values.reshape(-1, 1))
    df_latency_normalized = scaler.fit_transform(df_latency.values.reshape(-1, 1))
    
    data = np.concatenate([df_kpm_normalized, df_iperf_normalized, df_latency_normalized], axis=1)
    
    X = []
    y = []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y


def sdu_model_volume(df_kpm, df_iperf, df_latency, time_steps):
    df_kpm_selected = df_kpm[['DRB.RlcSduTransmittedVolumeUL']]#, '_time']]
    df_iperf_selected = df_iperf[['path_loss']]##, '_time']]
    df_latency_selected = df_latency[['time_latency']]##, '_time']]

    #df_kpm_selected['_time'] = pd.to_datetime(df_kpm_selected['_time'])
    #df_iperf_selected['_time'] = pd.to_datetime(df_iperf_selected['_time'])
    #df_latency_selected['_time'] = pd.to_datetime(df_latency_selected['_time'])

    df_kpm_numeric = df_kpm_selected.select_dtypes(include=np.number)
    print(df_iperf_selected)
    df_iperf_numeric = df_iperf_selected.select_dtypes(include=np.number)
    print(df_iperf_numeric)
    df_latency_numeric = df_latency_selected.select_dtypes(include=np.number)


    # Verificando se as colunas selecionadas não estão vazias
    if df_kpm_numeric.empty or df_iperf_numeric.empty or df_latency_numeric.empty:
        raise ValueError("Um ou mais DataFrames não contêm colunas numéricas")

    scaler = MinMaxScaler()
    df_kpm_normalized = scaler.fit_transform(df_kpm_numeric)
    df_iperf_normalized = scaler.fit_transform(df_iperf_numeric)
    df_latency_normalized = scaler.fit_transform(df_latency_numeric)
    
    data = np.concatenate([df_kpm_normalized, df_iperf_normalized, df_latency_normalized], axis=1)
    
    X = []
    y = []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y


def main():
    df_kpm = kpm_load()
    df_iperf = iperf_load()
    df_latency = latency_load()

    df_kpm.to_csv('./helpers/debug_df/kpms.csv')
    df_iperf.to_csv('./helpers/debug_df/iperf.csv')
    df_latency.to_csv('./helpers/debug_df/latency.csv')
    
    df_kpm['_time'] = pd.to_datetime(df_kpm['_time'])
    df_iperf['_time'] = pd.to_datetime(df_iperf['_time'])
    df_latency['_time'] = pd.to_datetime(df_latency['_time'])

    timestamps_filter = load_timestamps()

    filtered_df_kpm = filter_dataframe_by_test(df_kpm, 'mean12_stddev3', '1_test', timestamps_filter)
    filtered_df_iperf = filter_dataframe_by_test(df_iperf, 'mean12_stddev3', '1_test', timestamps_filter)
    filtered_df_latency = filter_dataframe_by_test(df_latency, 'mean12_stddev3', '1_test', timestamps_filter)


    kpm_plot(filtered_df_kpm, filtered_df_iperf, filtered_df_latency)

    #time_steps = 10

    #X, y = prepare_lstm_data(filtered_df_kpm, filtered_df_iperf, filtered_df_latency, time_steps)
    #print(filtered_df_latency)
    #sdu_model_volume(filtered_df_kpm, filtered_df_iperf, filtered_df_latency, time_steps)

if __name__ == "__main__":
    main()
