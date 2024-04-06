from sklearn.preprocessing import minmax_scale
import pandas as pd
import matplotlib.pyplot as plt
import srsRAN_data_treatment

def kpm_plot_single_test(df_kpm, df_iperf, df_latency):
    df_kpm = df_kpm[df_kpm['DRB.RlcSduTransmittedVolumeUL'] > 5]

    print(df_kpm.head())

    df_kpm_normalized = minmax_scale(df_kpm['DRB.RlcSduTransmittedVolumeUL'])
    df_iperf_normalized = minmax_scale(df_iperf['path_loss'])
    df_latency_normalized = minmax_scale(df_latency['time_latency'])

    plt.plot(df_kpm['_time'], df_kpm_normalized, '.', label='DRB.RlcSduTransmittedVolumeUL')
    plt.plot(df_iperf['_time'], df_iperf_normalized, '.', label='Path Loss')
    plt.plot(df_latency['_time'], df_latency_normalized, '.', label='Latency')

    plt.legend()
    plt.show()

def kpm_plot_all_tests_pl(df_kpm_list, df_iperf_list, df_latency_list, nr_tests):
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))  # 3 linhas, 2 colunas

    for test_i in range(nr_tests):
        df_kpm_normalized = minmax_scale(df_kpm_list[test_i]['DRB.RlcSduTransmittedVolumeUL'])
        df_iperf_normalized = minmax_scale(df_iperf_list[test_i]['path_loss'])
        df_latency_normalized = minmax_scale(df_latency_list[test_i]['time_latency'])

        row, col = divmod(test_i, 2)  # Calcula a posição do subplot
        axs[row, col].plot(df_kpm_list[test_i]['_time'], df_kpm_normalized, '.', label='DRB.RlcSduTransmittedVolumeUL')
        axs[row, col].plot(df_iperf_list[test_i]['_time'], df_iperf_normalized, '.', label='Path Loss')
        axs[row, col].plot(df_latency_list[test_i]['_time'], df_latency_normalized, '.', label='Latency')
        axs[row, col].set_title(f'Test {test_i + 1}')
        axs[row, col].legend()

    data_mean = srsRAN_data_treatment.get_mean_stddev(pd.concat(df_iperf_list)['path_loss'])[0]
    data_stddev = srsRAN_data_treatment.get_mean_stddev(pd.concat(df_iperf_list)['path_loss'])[1]
    fig.suptitle(f'Path Loss ({data_mean}, {data_stddev})')  # Define o título da figura
    plt.tight_layout()
    plt.show()