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

def get_df_multi_collection(df, tests_info_dict):
    df_list = []
    for distribution, tests in tests_info_dict.items():
        for test, timestamps in tests.items():
            start_timestamp = timestamps['start']
            end_timestamp = timestamps['end']
            filtered_df = filter_dataframe_by_timestamps(df, start_timestamp, end_timestamp)
            df_list.append(filtered_df)
    """
    counter = 0
    for df in df_list:
        counter = counter + df.shape[0]
    print("kpms => ", counter)
    """
    return df_list

def combine_dfs_by_test(df_kpm_list, df_iperf_list, df_latency_list):
    combined_dfs = []
    """total_kpms = 0

    for df in df_kpm_list:
        total_kpms = total_kpms + df.shape[0]
    print("total kpms => ", total_kpms)"""
    srsRAN_debug.write_list_csv(df_kpm_list, 'pre_kpm')
    srsRAN_debug.write_list_csv(df_iperf_list, 'pre_iperf')
    srsRAN_debug.write_list_csv(df_latency_list, 'pre_latency')
    for df_kpm, df_iperf, df_latency in zip(df_kpm_list, df_iperf_list, df_latency_list):
        df_combined = pd.merge(df_latency, df_iperf, on=['_time'], how='outer')
        df_combined = pd.merge(df_combined, df_kpm, on=['_time'], how='outer')
        combined_dfs.append(df_combined)
    return combined_dfs

def mean_by_tests(df_list, tests_info_dict):
    num_tests_dict = {}

    for key, value in tests_info_dict.items():
        num_tests = len(value)
        num_tests_dict[key] = num_tests

    print(num_tests_dict)

    dfs_by_category = {}
    start_idx = 0

    for category, num_tests in num_tests_dict.items():
        end_idx = start_idx + num_tests
        dfs_by_category[category] = df_list[start_idx:end_idx]
        start_idx = end_idx
    
    mean_values_by_category = {}

    for category, dfs in dfs_by_category.items():
        mean_path_loss_values = []
        mean_RlcSduTransmittedVolumeUL_values = []
        mean_UeThpUl_values = []
        mean_bitrate_values = []
        mean_jitter_values = []
        mean_lost_percentage_values = []
        mean_transfer_values = []
        
        for df in dfs:
            mean_path_loss_values.append(df['path_loss'].mean())
            mean_RlcSduTransmittedVolumeUL_values.append(df['DRB.RlcSduTransmittedVolumeUL'].mean())
            mean_UeThpUl_values.append(df['DRB.UEThpUl'].mean())
            mean_bitrate_values.append(df['bitrate'].mean())
            mean_jitter_values.append(df['jitter'].mean())
            mean_lost_percentage_values.append(df['lost_percentage'].mean())
            mean_transfer_values.append(df['transfer'].mean())
        
        mean_path_loss = sum(mean_path_loss_values) / len(mean_path_loss_values)
        mean_RlcSduTransmittedVolumeUL = sum(mean_RlcSduTransmittedVolumeUL_values) / len(mean_RlcSduTransmittedVolumeUL_values)
        mean_UeThpUl_values = sum(mean_UeThpUl_values) / len(mean_UeThpUl_values)
        mean_bitrate_values = sum(mean_bitrate_values) / len(mean_bitrate_values)
        mean_jitter_values = sum(mean_jitter_values) / len(mean_jitter_values)
        mean_lost_percentage_values = sum(mean_lost_percentage_values) / len(mean_lost_percentage_values)
        mean_transfer_values = sum(mean_transfer_values) / len(mean_transfer_values)

        mean_values_by_category[category] = {
            'mean_path_loss': mean_path_loss,
            'mean_RlcSduTransmittedVolumeUL': mean_RlcSduTransmittedVolumeUL,
            'mean_UeThpUl_values': mean_UeThpUl_values,
            'mean_bitrate_values': mean_bitrate_values,
            'mean_jitter_values': mean_jitter_values,
            'mean_lost_percentage_values': mean_lost_percentage_values,
            'mean_transfer_values': mean_transfer_values
        }

    # Print the mean values for each category
    for category, values in mean_values_by_category.items():
        print(f"Category: {category}")
        print(f"Mean path loss: {values['mean_path_loss']}")
        print(f"Mean DRB.RlcSduTransmittedVolumeUL: {values['mean_RlcSduTransmittedVolumeUL']}")
        print(f"Mean UeThp: {values['mean_UeThpUl_values']}")
        print(f"Mean bitrate: {values['mean_bitrate_values']}")    
        print(f"Mean jitter: {values['mean_jitter_values']}")    
        print(f"Mean lost_percentage: {values['mean_lost_percentage_values']}")
        print(f"Mean transfer_values: {values['mean_transfer_values']}")
        

def plots_custom_agg_by_test(df_kpm_list, df_iperf_list, df_latency_list):
    #TODO: Check if the values of beggining or end of the tests have influence in the final results, you know more or less what to expect..

    kpm_columns = ['DRB.PacketSuccessRateUlgNBUu', 'DRB.RlcPacketDropRateDl', 'DRB.RlcSduTransmittedVolumeDL', 'DRB.RlcSduTransmittedVolumeUL', 'DRB.UEThpDl', 'DRB.UEThpUl']
    # Just need the columns that we know that this will happen. The KPMs are always the last so we don't need to make anything to them
    agg_columns_iperf = ['bitrate', 'jitter', 'lost_percentage', 'path_loss', 'transfer']
    agg_columns_latency = ['time_latency']

    df_iperf_list = [df.assign(**{col: df[col].apply(pd.to_numeric, errors='coerce') for col in agg_columns_iperf}) for df in df_iperf_list]
    df_latency_list = [df.assign(**{col: df[col].apply(pd.to_numeric, errors='coerce') for col in agg_columns_latency}) for df in df_latency_list]

    accumulated_values = {col: [] for col in agg_columns_iperf+agg_columns_latency}
    counter = 0

    aggregated_times = []

    processed_dfs = []
    combined_df = combine_dfs_by_test(df_kpm_list, df_iperf_list, df_latency_list)
    srsRAN_debug.write_list_csv(combined_df, 'pre_processed')
    counter = 0
    for df in combined_df:
       # accumulated_values = {col: [] for col in agg_columns_iperf+agg_columns_latency}
        # aggregated_times = []
        
        for index, row in df.iterrows():
            if row[kpm_columns].notnull().any():
                counter += 1
                aggregated_times.append(row['_time'])
                for col in agg_columns_iperf+agg_columns_latency:
                    if accumulated_values[col]:
                        mean_values = sum(accumulated_values[col]) / len(accumulated_values[col])
                        df.loc[index, col] = round(mean_values,2)
                    else:
                        df.loc[index, col] = None
                accumulated_values = {col: [] for col in agg_columns_iperf+agg_columns_latency}
            else:
                for col in agg_columns_iperf+agg_columns_latency:
                    if pd.notnull(row[col]):
                        accumulated_values[col].append(row[col])
        
        df = df[df['_time'].isin(aggregated_times)]
        processed_dfs.append(df)

    processed_dfs = [df.dropna() for df in processed_dfs]

    srsRAN_debug.write_list_csv(processed_dfs, 'processed')

    return processed_dfs

    """df = df[df['_time'].isin(aggregated_times)]
    
    print("Count = ", counter)

    df = df.dropna()
    # Maybe after we can see if the number of ue has some impact via the correlations
    df = df.drop(columns=['ue_id', 'seq_nr'])

    #srsRAN_debug.write_csv(df, 'after_cust_agg')
    return df"""

    
def custom_agg_mean(df, keep_time):
    # TODO: Split by tests and just merge at the end ; Actual scenario --> All the tests together with this function
    # Hope that this is a temp function
    # Just need the columns that we know that this will happen. The KPMs are always the last so we don't need to make anything to them
    kpm_columns = ['DRB.PacketSuccessRateUlgNBUu', 'DRB.RlcPacketDropRateDl', 'DRB.RlcSduTransmittedVolumeDL', 'DRB.RlcSduTransmittedVolumeUL', 'DRB.UEThpDl', 'DRB.UEThpUl']
    agg_columns = ['bitrate', 'jitter', 'lost_percentage', 'path_loss', 'transfer', 'time_latency']
    df[agg_columns] = df[agg_columns].apply(pd.to_numeric, errors='coerce')

    accumulated_values = {col: [] for col in agg_columns}
    counter = 0

    if keep_time:
        aggregated_times = []

    for index, row in df.iterrows():
        if row[kpm_columns].notnull().any(): ### In the line only the KPMs must be present. The others can be obtain using the value of mean
            counter = counter + 1
            if keep_time:
                aggregated_times.append(row['_time'])
            for col in agg_columns:
                if accumulated_values[col]:
                    mean_values = sum(accumulated_values[col]) / len(accumulated_values[col])
                    df.loc[index, col] = mean_values
                else:
                    df.loc[index, col] = None
            accumulated_values = {col: [] for col in agg_columns}
        else:
            for col in agg_columns:
                if pd.notnull(row[col]):
                    accumulated_values[col].append(row[col])

    if keep_time:
        df = df[df['_time'].isin(aggregated_times)]
    
    print("Count = ", counter)

    df = df.dropna()
    # Maybe after we can see if the number of ue has some impact via the correlations
    if keep_time:
        df = df.drop(columns=['ue_id', 'seq_nr'])
    else:
        df = df.drop(columns=['ue_id', '_time', 'seq_nr'])

    #srsRAN_debug.write_csv(df, 'after_cust_agg')
    return df

def prepare_dfs_correlation(df_iperf, df_kpm, df_latency, keep_time):
    df_kpm = df_kpm[df_kpm['DRB.RlcSduTransmittedVolumeUL'] > 5] ### To stay only with lines when test is running
                                                                 ### This is just valid when you don't want to plot.
    print(df_kpm.shape[0])
    df_combined = pd.merge(df_iperf, df_kpm, on='_time', how='outer')
    df_combined = pd.merge(df_combined, df_latency, on='_time', how='outer')
    df_combined['bandwidth_required'] = df_combined['bandwidth_required'].apply(lambda x: float(x.replace('M', '')) * 1000000 if isinstance(x, str) else x)
    srsRAN_debug.write_csv(df_combined, 'concat')
    df_combined = custom_agg_mean(df_combined, keep_time)
    return df_combined



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