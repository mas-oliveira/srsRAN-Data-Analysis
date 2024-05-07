import os
import pandas as pd
import re

import srsRAN_data_treatment, srsRAN_debug, srsRAN_plots

"""
    AGGREGATE_PRBs
    This variable defines if you want to aggregate the DFs with 5,10,15,20 PRBs into one DF
    This will generate:
        df_kpms_prbs_agg.pkl
	    df_iperf_prbs_agg.pkl
	    df_latency_prbs_agg.pkl
"""
AGG_PRBS = False

"""
    MEAN_BY_PRB_AND_NOISE_AMPLITUDE
    This variable defines if you want to prepare data to analyze the KPMs variation according to
        multiple values of PRBs
    It will load:
        df_kpms_prbs_agg.pkl
	    df_iperf_prbs_agg.pkl
	    df_latency_prbs_agg.pkl
    And then the goal is to get a plot with the values of some KPMs per PRB
"""
MEAN_BY_PRB = True

"""
    MEAN_BY_PRB_AND_NOISE_AMPLITUDE
    This variable defines if you want to prepare data to analyze the KPMs variation according to
        multiple values of PRBs and Noise Amplitude.
    It will load:
        df_kpms_prbs_agg.pkl
	    df_iperf_prbs_agg.pkl
	    df_latency_prbs_agg.pkl
    And then the goal is to get a plot with the values of some KPMs per Noise Amplitude in all PRBs
"""
MEAN_BY_PRB_AND_NOISE_AMPLITUDE = True

PRB_TESTS = ["prb_25", "prb_52", "prb_79", "prb_106"]

"""
    Function: load_and_join_multi_prb
    This function is used to join the PRBs and add a new column with the prb value.
    Then it's possible to get in only one dataframe the results from multiple tests.
"""
def load_and_join_multi_prb(*files):
    print("FILES => ", files)
    combined_df = pd.DataFrame()

    for file_path in files:
        print(file_path)
        if os.path.exists(file_path):  
            pickle_df = pd.read_pickle(file_path)
            pickle_df['prb'] = int(re.search(r'prb_(\d+).pkl', file_path).group(1))
            combined_df = pd.concat([combined_df, pickle_df], ignore_index=True)
        else:
            print(f"File not found: {file_path}")
            print("Impossible to continue...")
            exit()

    return combined_df

def main():
    if AGG_PRBS:
        """
            In the variables kpms_prb, iperf_prb and latency_prb we have the results for all possible PRBs
            Then we'll need to get the noise_amplitude intervals to apply them to the other DFs
            It will generate pickles with the agg PRBs in the dataframes which all have also the noise_amplitude values
        """
        kpms_prb = load_and_join_multi_prb(*[f"./pickles/srsran_kpms/prbs/df_kpms_{test}.pkl" for test in PRB_TESTS])
        iperf_prb = load_and_join_multi_prb(*[f"./pickles/srsran_kpms/prbs/df_iperf_{test}.pkl" for test in PRB_TESTS])
        latency_prb = load_and_join_multi_prb(*[f"./pickles/srsran_kpms/prbs/df_latency_{test}.pkl" for test in PRB_TESTS])

        ### Short adjustment of float from gRPC
        iperf_prb['noise_amplitude'] = iperf_prb['noise_amplitude'].astype(float).round(1)

        ### In a first step instead of joining all the DFs the approach will be simple and the all the DFs will get the noise_amplitude values
        ###     which will be inserted according to the _time column in the kpms and latency DF.
        kpms_prb['collectStartTime'] = pd.to_datetime(kpms_prb['collectStartTime']).dt.tz_localize(None)
        iperf_prb['_time'] = pd.to_datetime(iperf_prb['_time']).dt.tz_localize(None)
        latency_prb['_time'] = pd.to_datetime(latency_prb['_time']).dt.tz_localize(None)

        kpms_prb = kpms_prb.sort_values('collectStartTime')
        iperf_prb = iperf_prb.sort_values('_time')
        latency_prb = latency_prb.sort_values('_time')

        kpms_prb = pd.merge_asof(kpms_prb, iperf_prb[['noise_amplitude', '_time']], left_on='collectStartTime', right_on='_time')
        latency_prb = pd.merge_asof(latency_prb, iperf_prb[['noise_amplitude', '_time']], left_on='_time', right_on='_time')

        ### Now it will be written on new DFs the new DFs of kpm and latency with PRB and Noise Amplitude values
        kpms_prb.to_pickle('./pickles/srsran_kpms/prbs/df_kpms_prbs_agg.pkl')
        iperf_prb.to_pickle('./pickles/srsran_kpms/prbs/df_iperf_prbs_agg.pkl')
        latency_prb.to_pickle('./pickles/srsran_kpms/prbs/df_latency_prbs_agg.pkl')

    if MEAN_BY_PRB:
        df_kpm = pd.read_pickle('./pickles/srsran_kpms/prbs/df_kpms_prbs_agg.pkl')
        df_iperf = pd.read_pickle('./pickles/srsran_kpms/prbs/df_iperf_prbs_agg.pkl')
        df_latency = pd.read_pickle('./pickles/srsran_kpms/prbs/df_latency_prbs_agg.pkl')
        
        av_dict_per_prb = srsRAN_data_treatment.get_metrics_per_prb(df_kpm, df_iperf, df_latency)
        srsRAN_plots.plot_metrics_av_per_prb(av_dict_per_prb)

    if MEAN_BY_PRB_AND_NOISE_AMPLITUDE:
        df_kpm = pd.read_pickle('./pickles/srsran_kpms/prbs/df_kpms_prbs_agg.pkl')
        df_iperf = pd.read_pickle('./pickles/srsran_kpms/prbs/df_iperf_prbs_agg.pkl')
        df_latency = pd.read_pickle('./pickles/srsran_kpms/prbs/df_latency_prbs_agg.pkl')

        av_dict_per_prb_and_an = srsRAN_data_treatment.get_metrics_per_prb_and_an(df_kpm, df_iperf, df_latency)
        srsRAN_plots.plot_metrics_av_per_prb_and_an(av_dict_per_prb_and_an)

if __name__ == "__main__":
    main()