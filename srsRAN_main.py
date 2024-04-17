import pandas as pd
import json as jason
import os

import srsRAN_data_treatment, srsRAN_plots, srsRAN_data_analysis, srsRAN_debug
 
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
ALL_TESTS = False

"""
    Plot variable
    SINGLE_TESTS
    This variable should be True if you want to plot just a specific test of a distribution
"""
SINGLE_TESTS = False

"""
    Plot variable
    MULTI_TEST
    This variable while on will work with ALL data from ALL tests present in the dataframe.
    The difference between this and the ALL_TESTS variable is that this one will join info of all the path
        loss distributions, while the ALL_TESTS will only care about the tests of a single path loss distribution.
"""
MULTI_TEST = False

"""
    Plot variable
    TEST_NR
    If you just want to plot a test you should specify here which one
"""
TEST_NR = '1_test'

"""
    Data selection / Plot variable
    PATH_LOSS_TESTS
    This variable is used to load tests with FIXED values of path loss
"""
PATH_LOSS_TESTS = ["fixed_pl_10", "fixed_pl_90"]

"""
    Data selection / Plot variable
    FIXED_TESTS
    If this variable is active (True) then the data collection will be picked according to the column
        test_number since you don't need anymore to "isolate" the tests according to the timestamps.
    After this upgrade there is a column named test_number where you don't need anymore to trim the dataset
        according to recorded timestamps on a json file 
"""
FIXED_TESTS = True

PLOTS = True
PRE_MLAI = True

"""
    Plot PRE ML/AI variable
    MEAN_PLOTS_TESTS
    This variable should be on if you want to see what's happening between the tests, most properly, to see 
        the impact of changes in path loss.
"""
MEAN_PLOTS_TESTS = True

CORRELATION = False

def kpm_load():
    return pd.read_pickle('./pickles/srsran_kpms/df_kpms.pkl')

def iperf_load():
    return pd.read_pickle('./pickles/srsran_kpms/df_iperf.pkl')

def latency_load():
    return pd.read_pickle('./pickles/srsran_kpms/df_latency.pkl')

def load_multi_tests(*files):
    print("FILES => ", files)
    combined_df = pd.DataFrame()  

    for file_path in files:
        print(file_path)
        if os.path.exists(file_path):  
            pickle_df = pd.read_pickle(file_path)
            combined_df = pd.concat([combined_df, pickle_df], ignore_index=True)
        else:
            print(f"File not found: {file_path}")
            print("Impossible to continue...")
            exit()

    return combined_df

def load_timestamps():
    with open('./helpers/timestamps.json', 'r') as file:
        timestamps = jason.load(file)
    return timestamps['srs_tests']

def main():
    #df_kpm = kpm_load()
    #df_iperf = iperf_load()
    #df_latency = latency_load()
    df_kpm = load_multi_tests(*[f"./pickles/srsran_kpms/df_kpms_{test}.pkl" for test in PATH_LOSS_TESTS])
    df_iperf = load_multi_tests(*[f"./pickles/srsran_kpms/df_iperf_{test}.pkl" for test in PATH_LOSS_TESTS])
    df_latency = load_multi_tests(*[f"./pickles/srsran_kpms/df_latency_{test}.pkl" for test in PATH_LOSS_TESTS])
    
    df_kpm['_time'] = pd.to_datetime(df_kpm['_time'])
    df_iperf['_time'] = pd.to_datetime(df_iperf['_time'])
    df_latency['_time'] = pd.to_datetime(df_latency['_time'])

    srsRAN_debug.write_csv(df_iperf['test_number'], 'iperf')

    timestamps_filter = load_timestamps()

    if PLOTS: 
        if ALL_TESTS:
            filtered_df_kpm = srsRAN_data_treatment.get_df_collection(df_kpm, PATH_LOSS_DISTRIBUTION, NR_TESTS, timestamps_filter)
            filtered_df_iperf = srsRAN_data_treatment.get_df_collection(df_iperf, PATH_LOSS_DISTRIBUTION, NR_TESTS, timestamps_filter)
            filtered_df_latency = srsRAN_data_treatment.get_df_collection(df_latency, PATH_LOSS_DISTRIBUTION, NR_TESTS, timestamps_filter)
            srsRAN_plots.kpm_plot_all_tests_pl(filtered_df_kpm, filtered_df_iperf, filtered_df_latency, NR_TESTS)
        elif SINGLE_TESTS:
            filtered_df_kpm = srsRAN_data_treatment.filter_dataframe_by_test(df_kpm, PATH_LOSS_DISTRIBUTION, TEST_NR, timestamps_filter)
            filtered_df_iperf = srsRAN_data_treatment.filter_dataframe_by_test(df_iperf, PATH_LOSS_DISTRIBUTION, TEST_NR, timestamps_filter)
            filtered_df_latency = srsRAN_data_treatment.filter_dataframe_by_test(df_latency, PATH_LOSS_DISTRIBUTION, TEST_NR, timestamps_filter)
            srsRAN_plots.kpm_plot_single_test(filtered_df_kpm, filtered_df_iperf, filtered_df_latency)
        if MULTI_TEST:
            # tests_info_dict is a dictionary that will contain the available distributions and the number of tests
            tests_info_dict = {distribution: {test: timestamps_filter[distribution][test] for test in timestamps_filter[distribution]} for distribution in timestamps_filter}
            print("############ MULTI TEST ############")
            print(tests_info_dict)
            print("############ END MULTI TEST ############")
            filtered_df_kpm = srsRAN_data_treatment.get_df_multi_collection(df_kpm, tests_info_dict)
            filtered_df_iperf = srsRAN_data_treatment.get_df_multi_collection(df_iperf, tests_info_dict)
            filtered_df_latency = srsRAN_data_treatment.get_df_multi_collection(df_latency, tests_info_dict)
            srsRAN_plots.kpm_plot_multi_tests_pl(filtered_df_kpm, filtered_df_iperf, filtered_df_latency)
        if FIXED_TESTS:
            tests_info_dict = srsRAN_data_treatment.get_tests_info_by_test_nr(df_iperf, df_latency)
            print("############ FIXED TEST ############")
            print(tests_info_dict)
            print("############ END FIXED TEST ############")            
            #splitted_df_kpm = srsRAN_data_treatment.split_df_by_test_number(df_kpm)
            #splitted_df_iperf = srsRAN_data_treatment.split_df_by_test_number(df_iperf)
            #splitted_df_latency = srsRAN_data_treatment.split_df_by_test_number(df_latency)
            
            splitted_df_kpm = srsRAN_data_treatment.get_df_multi_collection(df_kpm, tests_info_dict)
            splitted_df_iperf = srsRAN_data_treatment.get_df_multi_collection(df_iperf, tests_info_dict)
            splitted_df_latency = srsRAN_data_treatment.get_df_multi_collection(df_latency, tests_info_dict)

            #srsRAN_debug.write_list_csv(splitted_df_kpm, 'kpm_test_nr')
            srsRAN_plots.kpm_plot_multi_tests_pl(splitted_df_kpm, splitted_df_iperf, splitted_df_latency)

        

    if PRE_MLAI:   
        #### See if makes sense to change the name of prepare_dfs_correlation to prepare_dfs_numeric_pre_mlai
        if MEAN_PLOTS_TESTS:
            #print(df_kpm)
            #df_to_plot = srsRAN_data_treatment.prepare_dfs_correlation(df_iperf, df_kpm, df_latency, True)
            
            ### If doesn't need of timestamps.json to get the test times
            if FIXED_TESTS: 
                tests_info_dict = srsRAN_data_treatment.get_tests_info_by_test_nr(df_iperf, df_latency)
            else:
                tests_info_dict = {distribution: {test: timestamps_filter[distribution][test] for test in timestamps_filter[distribution]} for distribution in timestamps_filter}

            filtered_df_kpm = srsRAN_data_treatment.get_df_multi_collection(df_kpm, tests_info_dict)
            filtered_df_iperf = srsRAN_data_treatment.get_df_multi_collection(df_iperf, tests_info_dict)
            filtered_df_latency = srsRAN_data_treatment.get_df_multi_collection(df_latency, tests_info_dict)
            
            #print(filtered_df_kpm)
            df_agg_by_test = srsRAN_data_treatment.plots_custom_agg_by_test(filtered_df_kpm, filtered_df_iperf, filtered_df_latency)
            #print(df_agg_by_test)
            dict_to_plot = srsRAN_data_treatment.mean_by_tests(df_agg_by_test, tests_info_dict)
            print(dict_to_plot)
            srsRAN_plots.plot_data_all_categories(dict_to_plot)
            #srsRAN_debug.write_csv(df_to_plot, 'pre_plot_aiml')
            #print(df_to_plot)

        if CORRELATION:
            df_corr = srsRAN_data_treatment.prepare_dfs_correlation(df_iperf, df_kpm, df_latency, False)
            print("########## df_corr ##########")
            print(df_corr)
            """write_csv (df_corr, 'before_drop')
            df_corr = df_corr.dropna()
            write_csv (df_corr, 'after_drop')"""
            #print(df_corr)
            srsRAN_data_analysis.correlation_matrix(df_corr)

if __name__ == "__main__":
    main()
