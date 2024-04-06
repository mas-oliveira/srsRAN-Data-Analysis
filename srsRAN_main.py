import pandas as pd
import json as jason

import srsRAN_data_treatment, srsRAN_plots, srsRAN_data_analysis
 
"""
    Plot variable
    PATH_LOSS_DISTRIBUTION
    This variable should be filled according to the tests that you want to plot.
    The information of the tests is avaliable on timestamps.json
"""
PATH_LOSS_DISTRIBUTION = "mean12_stddev3"

"""
    Plot variable
    NR_TESTS => Number of tests
    The number of tests should be filled with the information according to number of tests
        of the path_loss_distribution that you want to plot. The default value is 6.
    The information of the tests is avaliable on timestamps.json 
"""
NR_TESTS = 6

"""
    Plot variable
    ALL_TESTS
    This variable choose if you want to do the plot of every tests or if you want to plot test by test.
"""
ALL_TESTS = True

"""
    Plot variable
    TEST_NR
    If you just want to plot a test you should specify here which one
"""
TEST_NR = '1_test'

PLOTS = False
MLAI = True

CORRELATION = True

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

def main():
    df_kpm = kpm_load()
    df_iperf = iperf_load()
    df_latency = latency_load()
    
    df_kpm['_time'] = pd.to_datetime(df_kpm['_time'])
    df_iperf['_time'] = pd.to_datetime(df_iperf['_time'])
    df_latency['_time'] = pd.to_datetime(df_latency['_time'])

    timestamps_filter = load_timestamps()

    if PLOTS: 
        if ALL_TESTS:
            filtered_df_kpm = srsRAN_data_treatment.get_df_collection(df_kpm, PATH_LOSS_DISTRIBUTION, NR_TESTS, timestamps_filter)
            filtered_df_iperf = srsRAN_data_treatment.get_df_collection(df_iperf, PATH_LOSS_DISTRIBUTION, NR_TESTS, timestamps_filter)
            filtered_df_latency = srsRAN_data_treatment.get_df_collection(df_latency, PATH_LOSS_DISTRIBUTION, NR_TESTS, timestamps_filter)
            srsRAN_plots.kpm_plot_all_tests_pl(filtered_df_kpm, filtered_df_iperf, filtered_df_latency, NR_TESTS)
        else:
            filtered_df_kpm = srsRAN_data_treatment.filter_dataframe_by_test(df_kpm, PATH_LOSS_DISTRIBUTION, TEST_NR, timestamps_filter)
            filtered_df_iperf = srsRAN_data_treatment.filter_dataframe_by_test(df_iperf, PATH_LOSS_DISTRIBUTION, TEST_NR, timestamps_filter)
            filtered_df_latency = srsRAN_data_treatment.filter_dataframe_by_test(df_latency, PATH_LOSS_DISTRIBUTION, TEST_NR, timestamps_filter)
            srsRAN_plots.kpm_plot_single_test(filtered_df_kpm, filtered_df_iperf, filtered_df_latency)

    if MLAI:
        if CORRELATION:
            df_corr = srsRAN_data_treatment.prepare_dfs_correlation(df_iperf, df_kpm, df_latency)
            """write_csv (df_corr, 'before_drop')
            df_corr = df_corr.dropna()
            write_csv (df_corr, 'after_drop')"""
            print(df_corr)
            srsRAN_data_analysis.correlation_matrix(df_corr)

if __name__ == "__main__":
    main()
