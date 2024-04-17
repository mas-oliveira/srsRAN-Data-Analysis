import srsRAN_data_treatment
import pandas as pd
import os

PATH_LOSS_TESTS = ["pl_90_f5"]
FILTER_PL_10_F5 = False
ADJUST_TEST_NR_PL_90_F5 = False

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

df_kpm = load_multi_tests(*[f"./pickles/srsran_kpms/df_kpms_{test}.pkl" for test in PATH_LOSS_TESTS])
df_iperf = load_multi_tests(*[f"./pickles/srsran_kpms/df_iperf_{test}.pkl" for test in PATH_LOSS_TESTS])
df_latency = load_multi_tests(*[f"./pickles/srsran_kpms/df_latency_{test}.pkl" for test in PATH_LOSS_TESTS])

if FILTER_PL_10_F5:
    df_kpm_filtered = df_kpm[df_kpm['test_number'].isin([150, 151, 153, 155])]
    df_iperf_filtered = df_iperf[df_iperf['test_number'].isin([150, 151, 153, 155])]
    df_latency_filtered = df_latency[df_latency['test_number'].isin([150, 151, 153, 155])]


    test = "pl_10_f5"
    df_kpm_filtered.to_pickle(f"./pickles/srsran_kpms/df_kpms_{test}.pkl")
    df_iperf_filtered.to_pickle(f"./pickles/srsran_kpms/df_iperf_{test}.pkl")
    df_latency_filtered.to_pickle(f"./pickles/srsran_kpms/df_latency_{test}.pkl")

if ADJUST_TEST_NR_PL_90_F5:
    print("############# PRE CHANGES #############")
    print(df_kpm)
    df_kpm['test_number'] = df_kpm['test_number'].replace({0: 1})
    print("############# POS CHANGES #############")
    print(df_kpm)
    test = "pl_90_f5"
    df_kpm.to_pickle(f"./pickles/srsran_kpms/df_kpms_{test}.pkl")