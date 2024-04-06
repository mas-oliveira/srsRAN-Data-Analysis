import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json as jason
from sklearn.preprocessing import minmax_scale, MinMaxScaler
"""from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split"""

KEYSIGHT_KPMS = ['CARR.WBCQIDist', 'DRB.AirIfDelayUl/s', 'DRB.PdcpSduVolumeUL_Filter', 'DRB.RlcDelayUl', 'DRB.UEThpUl', 'DRB.PreambleACell/s', 'RRC.ConnEstabSucc', 'RRC.ConnMean', 'RRU.PrbAvailUl', 'RRU.PrbTotUl', 'SM.PDUSessionSetupFail', 'SM.PDUSessionSetupReq', 'TB.ErrTotalNbrUl', 'TB.TotNbrUl']
#KPMS_TO_PLOT = ['DRB.AirIfDelayUl/s', 'DRB.UEThpUl', 'DRB.RlcDelayUl','RRU.PrbAvailUl', 'RRU.PrbTotUl']
KPMS_TO_PLOT=['RRU.PrbAvailUl']
PLOT_COLORS = ['green', 'red', 'blue', 'orange', 'purple']

def kpm_load():
    return pd.read_pickle('./pickles/keysight_kpms/df_kpms.pkl')

def load_timestamps():
    with open('timestamps.json', 'r') as file:
        timestamps = jason.load(file)
    return timestamps

def filter_dataframe_by_timestamps(df, start_timestamp, end_timestamp):
    return df[(df['_time'] >= start_timestamp) & (df['_time'] <= end_timestamp)]

def filter_dataframe_by_test(df, test_name, timestamps):
    start_timestamp = timestamps['keysight_tests'][test_name]['start']
    end_timestamp = timestamps['keysight_tests'][test_name]['end']
    return filter_dataframe_by_timestamps(df, start_timestamp, end_timestamp)

def kpm_plot(df_kpm):
    
    fig, ax1 = plt.subplots()

    for i, metric in enumerate(KPMS_TO_PLOT):
        normalized_metric = minmax_scale(df_kpm[metric])
        print(df_kpm[metric])
        ax1.plot(df_kpm['_time'], normalized_metric, '-',label=metric, color=PLOT_COLORS[i])

        if i > 0:
            ax = ax1.twinx()
            ax.plot(df_kpm['_time'], normalized_metric, '-', label=metric, color=PLOT_COLORS[i])
            ax.set_ylabel(metric)
            ax.tick_params('y')

    ax1.set_xlabel('Tempo')
    ax1.set_ylabel('Valores Normalizados')
    ax1.legend()

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
    
    df_kpm['_time'] = pd.to_datetime(df_kpm['_time'])
    
    print(df_kpm)
    
    timestamps_filter = load_timestamps()

    filtered_df_kpm = filter_dataframe_by_test(df_kpm, '6_test', timestamps_filter)

    print(filtered_df_kpm)
    kpm_plot(filtered_df_kpm)

    time_steps = 10

    #X, y = prepare_lstm_data(filtered_df_kpm, filtered_df_iperf, filtered_df_latency, time_steps)
    
    #sdu_model_volume(filtered_df_kpm, time_steps)

if __name__ == "__main__":
    main()
