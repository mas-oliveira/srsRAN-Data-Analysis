import pandas as pd
import srsRAN_data_treatment, srsRAN_debug, srsRAN_plots
import pickle
import numpy as np

### Test names with _treated after should only be used after treathing with TREATEMENT variable

#TEST_NAME = "multi_bitrate_and_noise_treated"
#TEST_NAME = "one_ue_latency"

#TEST_NAME = "one_ue_latency_noise"
#TEST_NAME = "one_ue_latency_noise_treated"  

#TEST_NAME = "one_ue_random_noise"
TEST_NAME = "one_ue_random_noise_treated"

#TEST_NAME = "two_ue_latency_noise"
#TEST_NAME = "two_ue_latency_noise_treated"

TEST_MULTI_BITRATE = True
TEST_MULTI_BITRATE_AND_NOISE = True 

BITRATE_AND_PRB = False
BITRATE_PRB_AND_AN = True

RANDOM_AN = True ### This variable is set to know if we have to select the tests of fixed An values or random an values (36 to 60)

TREATEMENT = False
PRB_INSERT = True
NOISE_INSERT = True
BITRATE_INSERT = True

SINGLE_UE = True # Same experiments but with just one UE (df_kpms_one_ue_latency_clean, df_iperf_one_ue_latency_clean, df_latency_one_ue_latency_clean)

def load_dataframes():
    df_kpm = pd.read_pickle(f'./pickles/srsran_kpms/df_kpms_{TEST_NAME}.pkl')
    df_iperf = pd.read_pickle(f'./pickles/srsran_kpms/df_iperf_{TEST_NAME}.pkl')
    df_latency = pd.read_pickle(f'./pickles/srsran_kpms/df_latency_{TEST_NAME}.pkl')
    return df_kpm, df_iperf, df_latency


#TEST_NUMBERS = [18,19,20,21,22,23,24,25,26]
def filter_tests_dataframes(df_kpm, df_iperf, df_latency):
    #df_kpm = df_kpm.query('test_number >= 18 and test_number <= 26')
    #df_iperf = df_iperf.query('test_number >= 18 and test_number <= 26')
    #df_latency = df_latency.query('test_number >= 18 and test_number <= 26')

    df_iperf.loc[df_iperf['test_number'] == 20, 'test_number'] = 19
    df_latency.loc[df_latency['test_number'] == 20, 'test_number'] = 19

    df_kpm['_time'] = df_kpm['_time'].astype(str)
    df_iperf['_time'] = df_iperf['_time'].astype(str)
    df_latency['_time'] = df_latency['_time'].astype(str)

    df_kpm['_time'] = df_kpm['_time'].str.replace(r'(\+\d{2}:\d{2}).*', r'\1', regex=True)
    df_iperf['_time'] = df_iperf['_time'].str.replace(r'(\+\d{2}:\d{2}).*', r'\1', regex=True)
    df_latency['_time'] = df_latency['_time'].str.replace(r'(\+\d{2}:\d{2}).*', r'\1', regex=True)

    df_kpm['_time'] = pd.to_datetime(df_kpm['_time'])
    df_iperf['_time'] = pd.to_datetime(df_iperf['_time'])
    df_latency['_time'] = pd.to_datetime(df_latency['_time'])

    start_time = pd.to_datetime('13:35:00', format='%H:%M:%S').time()
    end_time = pd.to_datetime('13:56:59', format='%H:%M:%S').time()

    df_iperf.loc[(df_iperf['test_number'] == 21) & (df_iperf['_time'].dt.time >= start_time) & (df_iperf['_time'].dt.time <= end_time), 'test_number'] = 20
    df_latency.loc[(df_latency['test_number'] == 21) & (df_latency['_time'].dt.time >= start_time) & (df_latency['_time'].dt.time <= end_time), 'test_number'] = 20

    return df_kpm, df_iperf, df_latency


def insert_bitrate_and_an_value(df_kpm, df_iperf, df_latency):
    df_kpm['_time'] = pd.to_datetime(df_kpm['_time'])
    df_latency['_time'] = pd.to_datetime(df_latency['_time'])
    df_iperf['_time'] = pd.to_datetime(df_iperf['_time'])
    
    df_iperf = df_iperf.sort_values(by='_time')

    for index, row in df_iperf.iterrows():
        test_number = row['test_number']
        bandwidth_required = row['bandwidth_required']
        noise_amplitude = row['noise_amplitude']
        timestamp = row['_time']
        
        df_kpm.loc[(df_kpm['test_number'] == test_number) & (df_kpm['_time'] == timestamp), 'bandwidth_required'] = bandwidth_required
        df_kpm.loc[(df_kpm['test_number'] == test_number) & (df_kpm['_time'] == timestamp), 'noise_amplitude'] = noise_amplitude
        
        df_latency.loc[(df_latency['test_number'] == test_number) & (df_latency['_time'] == timestamp), 'bandwidth_required'] = bandwidth_required
        df_latency.loc[(df_latency['test_number'] == test_number) & (df_latency['_time'] == timestamp), 'noise_amplitude'] = noise_amplitude

    return df_kpm, df_iperf, df_latency

def insert_bitrate_and_prb_value(df_kpm, df_iperf, df_latency):
    """df_kpm['_time'] = pd.to_datetime(df_kpm['_time'])
    df_latency['_time'] = pd.to_datetime(df_latency['_time'])
    df_iperf['_time'] = pd.to_datetime(df_iperf['_time'])
    
    df_iperf = df_iperf.sort_values(by='_time')"""

    for index, row in df_iperf.iterrows():
        test_number = row['test_number']
        bandwidth_required = row['bandwidth_required']
        prb = row['prb']
        timestamp = row['_time']
        
        df_kpm.loc[(df_kpm['test_number'] == test_number) & (df_kpm['_time'] == timestamp), 'bandwidth_required'] = bandwidth_required
        df_kpm.loc[(df_kpm['test_number'] == test_number) & (df_kpm['_time'] == timestamp), 'prb'] = prb
        
        df_latency.loc[(df_latency['test_number'] == test_number) & (df_latency['_time'] == timestamp), 'bandwidth_required'] = bandwidth_required
        df_latency.loc[(df_latency['test_number'] == test_number) & (df_latency['_time'] == timestamp), 'prb'] = prb

    pickle.dump(df_kpm, open(f'./pickles/srsran_kpms/df_kpms_{TEST_NAME}_treated.pkl', 'wb'))
    pickle.dump(df_iperf, open(f'./pickles/srsran_kpms/df_iperf_{TEST_NAME}_treated.pkl', 'wb'))
    pickle.dump(df_latency, open(f'./pickles/srsran_kpms/df_latency_{TEST_NAME}_treated.pkl', 'wb'))

    print(f"Dataframes tratados salvos em pickle com nome 'df_kpms_{TEST_NAME}_treated.pkl', 'df_iperf_{TEST_NAME}_treated.pkl' e 'df_latency_{TEST_NAME}_treated.pkl'.")


    return df_kpm, df_iperf, df_latency



def insert_bitrate_an_value_and_prb(df_kpm, df_iperf, df_latency):
    if 'ue_nr' not in df_latency.columns:
        df_latency = df_latency.rename(columns={'ue_id': 'ue_nr'})

    multi_ue = 'ue_nr' in df_kpm.columns

    df_kpm['_time'] = df_kpm['_time'].astype(str)
    df_iperf['_time'] = df_iperf['_time'].astype(str)
    df_latency['_time'] = df_latency['_time'].astype(str)

    df_kpm['_time'] = df_kpm['_time'].str.replace(r'(\+\d{2}:\d{2}).*', r'\1', regex=True)
    df_iperf['_time'] = df_iperf['_time'].str.replace(r'(\+\d{2}:\d{2}).*', r'\1', regex=True)
    df_latency['_time'] = df_latency['_time'].str.replace(r'(\+\d{2}:\d{2}).*', r'\1', regex=True)

    df_kpm['_time'] = pd.to_datetime(df_kpm['_time'])
    df_latency['_time'] = pd.to_datetime(df_latency['_time'])
    df_iperf['_time'] = pd.to_datetime(df_iperf['_time'])

    df_kpm['test_number'] = df_kpm['test_number'].astype(int)
    if multi_ue:
        df_kpm['ue_nr'] = df_kpm['ue_nr'].astype(int)

    df_latency['test_number'] = df_latency['test_number'].astype(int)
    if multi_ue:
        df_latency['ue_nr'] = df_latency['ue_nr'].astype(int)
    
    df_iperf = df_iperf.sort_values(by='_time')

    for index, row in df_iperf.iterrows():
        test_number = row['test_number']
        if multi_ue:
            ue_nr = row['ue_nr']
        bandwidth_required = row['bandwidth_required']
        noise_amplitude = row['noise_amplitude']
        prb = row['prb']
        timestamp = row['_time']

        if multi_ue:
            df_kpm.loc[(df_kpm['test_number'] == test_number) & (df_kpm['ue_nr'] == ue_nr) & (df_kpm['_time'] == timestamp), 'bandwidth_required'] = bandwidth_required
            df_kpm.loc[(df_kpm['test_number'] == test_number) & (df_kpm['ue_nr'] == ue_nr) & (df_kpm['_time'] == timestamp), 'noise_amplitude'] = noise_amplitude
            df_kpm.loc[(df_kpm['test_number'] == test_number) & (df_kpm['ue_nr'] == ue_nr) & (df_kpm['_time'] == timestamp), 'prb'] = prb
            df_latency.loc[(df_latency['test_number'] == test_number) & (df_latency['ue_nr'] == ue_nr) & (df_latency['_time'] == timestamp), 'bandwidth_required'] = bandwidth_required
            df_latency.loc[(df_latency['test_number'] == test_number) & (df_latency['ue_nr'] == ue_nr) & (df_latency['_time'] == timestamp), 'noise_amplitude'] = noise_amplitude
            df_latency.loc[(df_latency['test_number'] == test_number) & (df_latency['ue_nr'] == ue_nr) & (df_latency['_time'] == timestamp), 'prb'] = prb
        else:
            df_kpm.loc[(df_kpm['test_number'] == test_number) & (df_kpm['_time'] == timestamp), 'bandwidth_required'] = bandwidth_required
            df_kpm.loc[(df_kpm['test_number'] == test_number) & (df_kpm['_time'] == timestamp), 'noise_amplitude'] = noise_amplitude
            df_kpm.loc[(df_kpm['test_number'] == test_number) & (df_kpm['_time'] == timestamp), 'prb'] = prb
            df_latency.loc[(df_latency['test_number'] == test_number) & (df_latency['_time'] == timestamp), 'bandwidth_required'] = bandwidth_required
            df_latency.loc[(df_latency['test_number'] == test_number) & (df_latency['_time'] == timestamp), 'noise_amplitude'] = noise_amplitude
            df_latency.loc[(df_latency['test_number'] == test_number) & (df_latency['_time'] == timestamp), 'prb'] = prb
       

    pickle.dump(df_kpm, open(f'./pickles/srsran_kpms/df_kpms_{TEST_NAME}_treated.pkl', 'wb'))
    pickle.dump(df_iperf, open(f'./pickles/srsran_kpms/df_iperf_{TEST_NAME}_treated.pkl', 'wb'))
    pickle.dump(df_latency, open(f'./pickles/srsran_kpms/df_latency_{TEST_NAME}_treated.pkl', 'wb'))

    print(f"Dataframes tratados salvos em pickle com nome 'df_kpms_{TEST_NAME}_treated.pkl', 'df_iperf_{TEST_NAME}_treated.pkl' e 'df_latency_{TEST_NAME}_treated.pkl'.")

    return df_kpm, df_iperf, df_latency

def insert_bitrate_noise_and_prb_latency_dataframe (df_iperf, df_latency):
    if PRB_INSERT:
        prb_dict = df_iperf.groupby('test_number').first()['prb'].to_dict()
        
        df_latency['prb'] = df_latency['test_number'].map(prb_dict)

    if NOISE_INSERT:
        for test_number in df_iperf['test_number'].unique():
            df_iperf_test = df_iperf[df_iperf['test_number'] == test_number]
            df_latency_test = df_latency[df_latency['test_number'] == test_number]

            df_iperf_test = df_iperf_test.sort_values('_time')
            df_latency_test = df_latency_test.sort_values('_time')

            for i in range(len(df_iperf_test) - 1):
                start_time = df_iperf_test.iloc[i]['_time']
                end_time = df_iperf_test.iloc[i + 1]['_time']
                noise_amplitude = df_iperf_test.iloc[i]['noise_amplitude']

                mask = (df_latency_test['_time'] >= start_time) & (df_latency_test['_time'] < end_time)
                df_latency.loc[mask & (df_latency['test_number'] == test_number), 'noise_amplitude'] = noise_amplitude

            last_noise_amplitude = df_iperf_test.iloc[-1]['noise_amplitude']
            last_time = df_iperf_test.iloc[-1]['_time']
            mask = (df_latency_test['_time'] >= last_time)
            df_latency.loc[mask & (df_latency['test_number'] == test_number), 'noise_amplitude'] = last_noise_amplitude

    if BITRATE_INSERT:
        df_iperf['ue_nr'] = df_iperf['ue_nr'].astype(int)
        df_latency['ue_nr'] = df_latency['ue_nr'].astype(int)

        bandwidth_dict = df_iperf.set_index(['test_number', 'ue_nr'])['bandwidth_required'].to_dict()
        print("bandwidth_dict:", bandwidth_dict)

        def map_bandwidth(row):
            key = (row['test_number'], row['ue_nr'])
            value = bandwidth_dict.get(key)
            if value is None:
                print(f"Warning: No bandwidth_required for key {key}")
            return value

        df_latency['bandwidth_required'] = df_latency.apply(map_bandwidth, axis=1)


    pickle.dump(df_iperf, open(f'./pickles/srsran_kpms/df_iperf_{TEST_NAME}.pkl', 'wb'))
    pickle.dump(df_latency, open(f'./pickles/srsran_kpms/df_latency_{TEST_NAME}.pkl', 'wb'))
    return df_latency

def adjust_time_column(df_kpm, df_iperf, df_latency):
    df_kpm['_time'] = df_kpm['_time'].astype(str)
    df_iperf['_time'] = df_iperf['_time'].astype(str)
    df_latency['_time'] = df_latency['_time'].astype(str)

    df_kpm['_time'] = df_kpm['_time'].str.replace(r'(\+\d{2}:\d{2}).*', r'\1', regex=True)
    df_iperf['_time'] = df_iperf['_time'].str.replace(r'(\+\d{2}:\d{2}).*', r'\1', regex=True)
    df_latency['_time'] = df_latency['_time'].str.replace(r'(\+\d{2}:\d{2}).*', r'\1', regex=True)

    df_kpm['_time'] = pd.to_datetime(df_kpm['_time'])
    df_iperf['_time'] = pd.to_datetime(df_iperf['_time'])
    df_latency['_time'] = pd.to_datetime(df_latency['_time'])

    return df_kpm, df_iperf, df_latency


BAD_TESTS = [12, 20]
def remove_bad_test(df_kpm, df_iperf, df_latency):
    df_kpm = df_kpm[~df_kpm['test_number'].isin(BAD_TESTS)]
    df_iperf = df_iperf[~df_iperf['test_number'].isin(BAD_TESTS)]
    df_latency = df_latency[~df_latency['test_number'].isin(BAD_TESTS)]
    
    return df_kpm, df_iperf, df_latency

### aux function to make the name of plots
def get_test_info(df_latency):
    test_dict = {}
    
    for _, row in df_latency.iterrows():
        test_number = row['test_number']
        
        if test_number not in test_dict:
            test_dict[test_number] = {
                'bandwidth_required': row['bandwidth_required'],
                'prb': int(row['prb'])
            }
    
    return test_dict

def combine_info_and_latencies(dict_info, dict_latencies):
    combined_data = {}

    for test_number, info in dict_info.items():
        key = (info['bandwidth_required'], info['prb'])
        if key not in combined_data:
            combined_data[key] = {}

        test_id = f'test_{test_number}'
        if test_id in dict_latencies:
            for noise, latency in dict_latencies[test_id].items():
                if noise not in combined_data[key]:
                    combined_data[key][noise] = np.array([], dtype=float)
                combined_data[key][noise] = np.append(combined_data[key][noise], latency)

    return combined_data

def main():
    df_kpm, df_iperf, df_latency = load_dataframes()
    if SINGLE_UE is False:
        if TEST_MULTI_BITRATE:
            df_kpm = df_kpm[df_kpm['ue_nr'].isna()]
            df_kpm['ue_nr'].fillna(1, inplace=True)
            df_kpm, df_iperf, df_latency = filter_tests_dataframes(df_kpm, df_iperf, df_latency)
            df_kpm, df_iperf, df_latency = insert_bitrate_and_an_value(df_kpm, df_iperf, df_latency)
            av_dict_per_prb_and_an = srsRAN_data_treatment.get_metrics_per_bitrate_and_an(df_kpm, df_iperf, df_latency)
            print(av_dict_per_prb_and_an)
            srsRAN_plots.plot_metrics_av_per_bitrate_and_an(av_dict_per_prb_and_an)
        elif TEST_MULTI_BITRATE_AND_NOISE:
            if TREATEMENT:
                #df_kpm, df_iperf, df_latency = remove_bad_test(df_kpm, df_iperf, df_latency) => could also be done on aux_treat... in some cases
                df_kpm, df_iperf, df_latency = insert_bitrate_an_value_and_prb(df_kpm, df_iperf, df_latency)

            else:
                ### Uncomment to generate metrics plots
                #av_dict_per_prb_bitrate_and_an = srsRAN_data_treatment.get_metrics_per_bitrate_an_and_prb(df_kpm, df_iperf, df_latency, RANDOM_AN)
                #print(av_dict_per_prb_bitrate_and_an)
                #srsRAN_plots.plot_metrics_av_per_bitrate_an_prb(av_dict_per_prb_bitrate_and_an)

                dict_latencies = srsRAN_data_treatment.generate_latency_arrays(df_latency)
                #print(dict_latencies)

                #srsRAN_plots.plot_latencies_per_test(dict_latencies)

                dict_info = get_test_info(df_latency)
                dict_latencies = srsRAN_data_treatment.generate_latency_arrays_with_noise(df_latency)
                srsRAN_plots.plot_latencies_box_plots_per_test(dict_latencies, dict_info)

    elif SINGLE_UE is True:
        if TEST_MULTI_BITRATE:

            if TREATEMENT:
                if BITRATE_AND_PRB:
                    df_kpm, df_iperf, df_latency = adjust_time_column(df_kpm, df_iperf, df_latency)
                    df_kpm, df_iperf, df_latency = insert_bitrate_and_prb_value(df_kpm, df_iperf, df_latency)
                    srsRAN_debug.write_csv(df_kpm, 'df_kpm_debug')
                    srsRAN_debug.write_csv(df_iperf, 'df_iperf_debug')
                    srsRAN_debug.write_csv(df_latency, 'df_latency_debug')
                if BITRATE_PRB_AND_AN:
                    df_kpm, df_iperf, df_latency = adjust_time_column(df_kpm, df_iperf, df_latency)
                    df_kpm, df_iperf, df_latency = insert_bitrate_an_value_and_prb(df_kpm, df_iperf, df_latency)
                    
                    srsRAN_debug.write_csv(df_kpm, 'df_kpm_debug')
                    srsRAN_debug.write_csv(df_iperf, 'df_iperf_debug')
                    srsRAN_debug.write_csv(df_latency, 'df_latency_debug')
            else:
                if BITRATE_AND_PRB:
                    av_dict_per_prb_and_an = srsRAN_data_treatment.get_metrics_per_bitrate_and_prb(df_kpm, df_iperf, df_latency)
                    print(av_dict_per_prb_and_an)
                    srsRAN_plots.plot_metrics_av_per_bitrate_and_prb(av_dict_per_prb_and_an)
                elif BITRATE_PRB_AND_AN:
                    av_dict_per_prb_bitrate_and_an = srsRAN_data_treatment.get_metrics_per_bitrate_an_and_prb(df_kpm, df_iperf, df_latency, RANDOM_AN)
                    #print(av_dict_per_prb_bitrate_and_an)

                    #srsRAN_plots.plot_metrics_av_per_bitrate_an_prb(av_dict_per_prb_bitrate_and_an, RANDOM_AN)
                    
                    ### Just to plot latency
                    #dict_latencies = srsRAN_data_treatment.generate_latency_arrays(df_latency)
                    #srsRAN_plots.plot_latencies_per_test(dict_latencies)

                    ### Plot latency with noise
                    dict_info = get_test_info(df_latency)
                    dict_latencies = srsRAN_data_treatment.generate_latency_arrays_with_noise(df_latency, RANDOM_AN)
                    agg_dict_latencies = combine_info_and_latencies(dict_info, dict_latencies)

                    #srsRAN_plots.plot_latencies_box_plots_per_test(dict_latencies, dict_info, RANDOM_AN)
                    srsRAN_plots.plot_agg_latencies_box_plots(agg_dict_latencies)


if __name__ == "__main__":
    main()