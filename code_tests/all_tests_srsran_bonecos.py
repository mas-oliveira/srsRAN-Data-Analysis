import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json as jason
from sklearn.preprocessing import minmax_scale, MinMaxScaler
"""from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split"""

### CRIAR MAIS DEFS PARA FICAR MAIS RAPIDO MUDAR DE TESTES
### METER PRONTO PARA METER EM VARIAS REDES NEURONAIS DEPOIS DE LER O QUE DA PARA USAR PARA INFERIR COISAS SOBRE OS DADOS QUE TENHO

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

def filter_dataframe_by_test(df, test_conditions, nr_tests, timestamps):
    df_list = []
    for test_i in range(1, nr_tests + 1):
        start_timestamp = timestamps[test_conditions][f'{test_i}_test']['start']
        end_timestamp = timestamps[test_conditions][f'{test_i}_test']['end']
        df_list.append(filter_dataframe_by_timestamps(df, start_timestamp, end_timestamp))
    return df_list

def kpm_plot(df_kpm, df_iperf, df_latency, nr_tests):
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))  # 3 linhas, 2 colunas

    for test_i in range(nr_tests):
        df_kpm_normalized = minmax_scale(df_kpm[test_i]['DRB.RlcSduTransmittedVolumeUL'])
        df_iperf_normalized = minmax_scale(df_iperf[test_i]['path_loss'])
        df_latency_normalized = minmax_scale(df_latency[test_i]['time_latency'])

        row, col = divmod(test_i, 2)  # Calcula a posição do subplot
        axs[row, col].plot(df_kpm[test_i]['_time'], df_kpm_normalized, '.', label='DRB.RlcSduTransmittedVolumeUL')
        axs[row, col].plot(df_iperf[test_i]['_time'], df_iperf_normalized, '.', label='Path Loss')
        axs[row, col].plot(df_latency[test_i]['_time'], df_latency_normalized, '.', label='Latency')
        axs[row, col].set_title(f'Test {test_i + 1}')
        axs[row, col].legend()

    ### DINAMICO
    fig.suptitle('Path Loss mean=12 stddev=3')  # Define o título da figura
    plt.tight_layout()
    plt.show()


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

    nr_tests = 6

    filtered_df_kpm = filter_dataframe_by_test(df_kpm, 'mean12_stddev3', nr_tests, timestamps_filter)
    filtered_df_iperf = filter_dataframe_by_test(df_iperf, 'mean12_stddev3', nr_tests, timestamps_filter)
    filtered_df_latency = filter_dataframe_by_test(df_latency, 'mean12_stddev3', nr_tests, timestamps_filter)

    kpm_plot(filtered_df_kpm, filtered_df_iperf, filtered_df_latency, nr_tests)

    #time_steps = 10

    #X, y = prepare_lstm_data(filtered_df_kpm, filtered_df_iperf, filtered_df_latency, time_steps)
    #print(filtered_df_latency)
    #sdu_model_volume(filtered_df_kpm, filtered_df_iperf, filtered_df_latency, time_steps)

if __name__ == "__main__":
    main()
