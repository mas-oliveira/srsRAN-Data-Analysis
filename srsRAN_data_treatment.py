from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import srsRAN_debug

def filter_dataframe_by_timestamps(df, start_timestamp, end_timestamp):
    return df[(df['_time'] >= start_timestamp) & (df['_time'] <= end_timestamp)]

def filter_dataframe_by_test(df, test_conditions, test_nr, timestamps):
    start_timestamp = timestamps[test_conditions][test_nr]['start']
    end_timestamp = timestamps[test_conditions][test_nr]['end']
    return filter_dataframe_by_timestamps(df, start_timestamp, end_timestamp)

def get_mean_stddev(df_col):
    return round(df_col.mean(), 2), round(df_col.std(), 2)

def get_df_collection(df, pl_dist, nr_tests, timestamps):
    df_list = []
    for test_i in range(1, nr_tests + 1):
        start_timestamp = timestamps[pl_dist][f'{test_i}_test']['start']
        end_timestamp = timestamps[pl_dist][f'{test_i}_test']['end']
        df_list.append(filter_dataframe_by_timestamps(df, start_timestamp, end_timestamp))
    return df_list

def prepare_dfs_correlation(df_kpm, df_iperf, df_latency):
    df_combined = pd.merge(df_kpm, df_iperf, on='_time', how='outer')
    df_combined = pd.merge(df_combined, df_latency, on='_time', how='outer')
    srsRAN_debug.write_csv(df_combined, 'concat')
     ### TROCAR ISTO POR UMA FORMA DE FICAR SO COM OS NUMEROS; ACHO QUE NAO FUNCIONA SO ASSIM
    only_numeric_cols = df_combined.select_dtypes(include='number')##.drop(columns=['_field'], inplace=True)
    return only_numeric_cols



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