import pandas as pd
import numpy as np

TEST_NAME = "one_ue_latency_noise"
FAULTY_TEST_NAME = "one_ue"

WITHOUT_NOISE = False
WITH_NOISE = True

def load_dataframes():
    df_kpm = pd.read_pickle(f'./pickles/srsran_kpms/df_kpms_{TEST_NAME}.pkl')
    df_iperf = pd.read_pickle(f'./pickles/srsran_kpms/df_iperf_{TEST_NAME}.pkl')
    df_latency = pd.read_pickle(f'./pickles/srsran_kpms/df_latency_{TEST_NAME}.pkl')
    return df_kpm, df_iperf, df_latency

def load_faulty_dataframe():
    return pd.read_pickle(f'./pickles/srsran_kpms/df_latency_{FAULTY_TEST_NAME}.pkl')

def filter_and_increment_test_number(df_latency):
    # Filtrar para remover ocorrências onde seq_nr está entre 1 e 4
    df_latency_filtered = df_latency[~df_latency['seq_nr'].isin(["1", "2", "3", "4"])]
    
    test_number = 19
    for index, row in df_latency_filtered.iterrows():
        if row['seq_nr'] == "5":
            test_number += 1
        df_latency_filtered.at[index, 'test_number'] = test_number

    df_latency_filtered['test_number'] = df_latency_filtered['test_number'].replace({24: 25, 25: 26, 26: 27})

    
    return df_latency_filtered

def trim_dataframes(df_kpm, df_iperf, df_latency):
    unique_test_numbers = set(df_kpm['test_number']).intersection(set(df_iperf['test_number'])).intersection(set(df_latency['test_number']))
    
    df_kpm_trimmed = pd.DataFrame()
    df_iperf_trimmed = pd.DataFrame()
    df_latency_trimmed = pd.DataFrame()
    
    for test_number in unique_test_numbers:
        df_kpm_temp = df_kpm[df_kpm['test_number'] == test_number]
        df_iperf_temp = df_iperf[df_iperf['test_number'] == test_number]
        df_latency_temp = df_latency[df_latency['test_number'] == test_number]

        df_kpm_temp['_time'] = df_kpm_temp['_time'].astype(str)
        df_iperf_temp['_time'] = df_iperf_temp['_time'].astype(str)
        df_latency_temp['_time'] = df_latency_temp['_time'].astype(str)

        df_kpm_temp['_time'] = df_kpm_temp['_time'].str.replace(r'(\+\d{2}:\d{2}).*', r'\1', regex=True)
        df_iperf_temp['_time'] = df_iperf_temp['_time'].str.replace(r'(\+\d{2}:\d{2}).*', r'\1', regex=True)
        df_latency_temp['_time'] = df_latency_temp['_time'].str.replace(r'(\+\d{2}:\d{2}).*', r'\1', regex=True)

        df_kpm_temp['_time'] = pd.to_datetime(df_kpm_temp['_time'])
        df_iperf_temp['_time'] = pd.to_datetime(df_iperf_temp['_time'])
        df_latency_temp['_time'] = pd.to_datetime(df_latency_temp['_time'])

        min_time = max(df_kpm_temp['_time'].min(), df_iperf_temp['_time'].min(), df_latency_temp['_time'].min())
        max_time = min(df_kpm_temp['_time'].max(), df_iperf_temp['_time'].max(), df_latency_temp['_time'].max())
        
        df_kpm_temp_trimmed = df_kpm_temp[(df_kpm_temp['_time'] >= min_time) & (df_kpm_temp['_time'] <= max_time)]
        df_iperf_temp_trimmed = df_iperf_temp[(df_iperf_temp['_time'] >= min_time) & (df_iperf_temp['_time'] <= max_time)]
        df_latency_temp_trimmed = df_latency_temp[(df_latency_temp['_time'] >= min_time) & (df_latency_temp['_time'] <= max_time)]
        
        df_kpm_trimmed = pd.concat([df_kpm_trimmed, df_kpm_temp_trimmed])
        df_iperf_trimmed = pd.concat([df_iperf_trimmed, df_iperf_temp_trimmed])
        df_latency_trimmed = pd.concat([df_latency_trimmed, df_latency_temp_trimmed])
    
    return df_kpm_trimmed, df_iperf_trimmed, df_latency_trimmed


BAD_TESTS_WITHOUT_NOISE = [1, 7, 17, 19, 23, 24]
BAD_TESTS_WITH_NOISE = [1, 2 , 3, 17, 19, 21, 32]


def main():
    df_kpm, df_iperf, df_latency = load_dataframes()

    if WITHOUT_NOISE:
        df_latency_bad = load_faulty_dataframe()
        df_latency_bad = filter_and_increment_test_number(df_latency_bad)
        
        df_latency = pd.concat([df_latency, df_latency_bad]).sort_values('_time')

        df_kpm = df_kpm[~df_kpm['test_number'].isin(BAD_TESTS_WITHOUT_NOISE)]
        df_iperf = df_iperf[~df_iperf['test_number'].isin(BAD_TESTS_WITHOUT_NOISE)]
        df_latency = df_latency[~df_latency['test_number'].isin(BAD_TESTS_WITHOUT_NOISE)]

        print("Test number counts ordered in df_kpm:")
        test_number_counts_kpm = df_kpm['test_number'].value_counts().sort_index()
        print(test_number_counts_kpm)

        print("Test number counts ordered in df_iperf:")
        test_number_counts_iperf = df_iperf['test_number'].value_counts().sort_index()
        print(test_number_counts_iperf)

        print("Test number counts ordered in df_latency:")
        test_number_counts_latency = df_latency['test_number'].value_counts().sort_index()
        print(test_number_counts_latency)
        
        df_kpm, df_iperf, df_latency = trim_dataframes(df_kpm, df_iperf, df_latency)

        print("Test number counts ordered in df_kpm:")
        test_number_counts_kpm = df_kpm['test_number'].value_counts().sort_index()
        print(test_number_counts_kpm)

        print("Test number counts ordered in df_iperf:")
        test_number_counts_iperf = df_iperf['test_number'].value_counts().sort_index()
        print(test_number_counts_iperf)

        print("Test number counts ordered in df_latency:")
        test_number_counts_latency = df_latency['test_number'].value_counts().sort_index()
        print(test_number_counts_latency)

        df_kpm.to_pickle('./pickles/srsran_kpms/df_kpms_one_ue_latency.pkl')
        df_iperf.to_pickle('./pickles/srsran_kpms/df_iperf_one_ue_latency.pkl')
        df_latency.to_pickle('./pickles/srsran_kpms/df_latency_one_ue_latency.pkl')

    if WITH_NOISE:
        df_kpm = df_kpm[~df_kpm['test_number'].isin(BAD_TESTS_WITH_NOISE)]
        df_iperf = df_iperf[~df_iperf['test_number'].isin(BAD_TESTS_WITH_NOISE)]
        df_latency = df_latency[~df_latency['test_number'].isin(BAD_TESTS_WITH_NOISE)]

        print("Test number counts ordered in df_kpm:")
        test_number_counts_kpm = df_kpm['test_number'].value_counts().sort_index()
        print(test_number_counts_kpm)

        print("Test number counts ordered in df_iperf:")
        test_number_counts_iperf = df_iperf['test_number'].value_counts().sort_index()
        print(test_number_counts_iperf)

        print("Test number counts ordered in df_latency:")
        test_number_counts_latency = df_latency['test_number'].value_counts().sort_index()
        print(test_number_counts_latency)

        df_kpm, df_iperf, df_latency = trim_dataframes(df_kpm, df_iperf, df_latency)

        print("Test number counts ordered in df_kpm:")
        test_number_counts_kpm = df_kpm['test_number'].value_counts().sort_index()
        print(test_number_counts_kpm)

        print("Test number counts ordered in df_iperf:")
        test_number_counts_iperf = df_iperf['test_number'].value_counts().sort_index()
        print(test_number_counts_iperf)

        print("Test number counts ordered in df_latency:")
        test_number_counts_latency = df_latency['test_number'].value_counts().sort_index()
        print(test_number_counts_latency)

        df_kpm.to_pickle('./pickles/srsran_kpms/df_kpms_one_ue_latency_noise.pkl')
        df_iperf.to_pickle('./pickles/srsran_kpms/df_iperf_one_ue_latency_noise.pkl')
        df_latency.to_pickle('./pickles/srsran_kpms/df_latency_one_ue_latency_noise.pkl')

if __name__ == "__main__":
    main()
