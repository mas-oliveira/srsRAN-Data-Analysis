from sklearn.preprocessing import minmax_scale
import pandas as pd
import matplotlib.pyplot as plt
import srsRAN_data_treatment

def kpm_plot_single_test(df_kpm, df_iperf, df_latency):
    df_kpm = df_kpm[df_kpm['DRB.RlcSduTransmittedVolumeUL'] > 5]

    df_kpm_normalized = minmax_scale(df_kpm['DRB.RlcSduTransmittedVolumeUL'])
    df_iperf_normalized = minmax_scale(df_iperf['path_loss'])
    df_latency_normalized = minmax_scale(df_latency['time_latency'])

    plt.plot(df_kpm['_time'], df_kpm_normalized, '.', label='DRB.RlcSduTransmittedVolumeUL')
    plt.plot(df_iperf['_time'], df_iperf_normalized, '.', label='Path Loss')
    plt.plot(df_latency['_time'], df_latency_normalized, '.', label='Latency')

    plt.legend()
    plt.show()

def kpm_plot_all_tests_pl(df_kpm_list, df_iperf_list, df_latency_list, nr_tests):
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))  

    for test_i in range(nr_tests):
        df_kpm_normalized = minmax_scale(df_kpm_list[test_i]['DRB.RlcSduTransmittedVolumeUL'])
        df_iperf_normalized = minmax_scale(df_iperf_list[test_i]['path_loss'])
        df_latency_normalized = minmax_scale(df_latency_list[test_i]['time_latency'])

        row, col = divmod(test_i, 2) 
        axs[row, col].plot(df_kpm_list[test_i]['_time'], df_kpm_normalized, '.', label='DRB.RlcSduTransmittedVolumeUL')
        axs[row, col].plot(df_iperf_list[test_i]['_time'], df_iperf_normalized, '.', label='Path Loss')
        axs[row, col].plot(df_latency_list[test_i]['_time'], df_latency_normalized, '.', label='Latency')
        axs[row, col].set_title(f'Test {test_i + 1}')
        axs[row, col].legend()

    data_mean = srsRAN_data_treatment.get_mean_stddev(pd.concat(df_iperf_list)['path_loss'])[0]
    data_stddev = srsRAN_data_treatment.get_mean_stddev(pd.concat(df_iperf_list)['path_loss'])[1]
    fig.suptitle(f'Path Loss ({data_mean}, {data_stddev})') 
    plt.tight_layout()
    plt.show()

def kpm_plot_multi_tests_pl(df_kpm_list, df_iperf_list, df_latency_list):
    plt.figure(figsize=(10, 6))
    all_data_kpms = []
    all_times_kpms = []
    all_data_iperf = []
    all_times_iperf = []
    all_data_latency = []
    all_times_latency = []

    for df in df_kpm_list:
        filtered_values = df['DRB.RlcSduTransmittedVolumeUL'][df['DRB.RlcSduTransmittedVolumeUL'] > 5]
        filtered_times = pd.to_datetime(df['_time'][df['DRB.RlcSduTransmittedVolumeUL'] > 5])
        all_data_kpms.extend(filtered_values.tolist())
        all_times_kpms.extend(filtered_times.tolist())
    
    for df in df_iperf_list:
        all_data_iperf.extend(df['path_loss'].tolist())
        all_times_iperf.extend(pd.to_datetime(df['_time']).tolist())

    for df in df_latency_list:
        all_data_latency.extend(df['time_latency'].tolist())
        all_times_latency.extend(pd.to_datetime(df['_time']).tolist())

    df_kpm_normalized = minmax_scale(all_data_kpms)
    df_iperf_normalized = minmax_scale(all_data_iperf)
    df_latency_normalized = minmax_scale(all_data_latency)

    plt.plot(all_times_kpms, df_kpm_normalized, '.', label='DRB.RlcSduTransmittedVolumeUL')
    plt.plot(all_times_iperf, df_iperf_normalized, '.', label='Path Loss')
    plt.plot(all_times_latency, df_latency_normalized, '.', label='Latency')

    plt.legend()
    plt.show()