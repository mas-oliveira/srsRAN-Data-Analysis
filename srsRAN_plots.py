from sklearn.preprocessing import minmax_scale
import pandas as pd
import matplotlib.pyplot as plt
import srsRAN_data_treatment
import numpy as np
import os

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
        print(df['DRB.RlcSduTransmittedVolumeUL'])
        df['DRB.RlcSduTransmittedVolumeUL'] = df['DRB.RlcSduTransmittedVolumeUL'].astype(int)
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

def plot_data_all_categories(normalized_values_by_category):
    num_categories = len(normalized_values_by_category)
    if num_categories == 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['palegreen', 'lightskyblue']  

        category_labels = list(normalized_values_by_category.keys())
        categories_data = list(normalized_values_by_category.values())

        bar_width = 0.35
        bar_positions = np.arange(len(categories_data[0]))

        for i, (category, values) in enumerate(zip(category_labels, categories_data)):
            ax.bar(bar_positions + i * bar_width, values.values(), bar_width, label=f'{category}', color=colors[i])

        ax.set_title('Normalized values for both categories')
        ax.set_ylabel('Normalized values')
        ax.set_xticks(bar_positions + bar_width / 2)
        ax.set_xticklabels(values.keys())
        ax.tick_params(axis='x', rotation=25)  
        ax.set_ylim([0, 1])
        ax.legend()
        plt.tight_layout()
        plt.show()
    else:
        num_cols = 2  
        num_rows = (num_categories + 1) // 2  

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6*num_rows))

        for i, (category, values) in enumerate(normalized_values_by_category.items()):
            row = i // num_cols
            col = i % num_cols
            ax = axes[row, col] if num_rows > 1 else axes[col]  

            ax.bar(values.keys(), values.values())
            ax.set_title(f'Category: {category}')
            ax.set_ylabel('Normalized values')
            ax.tick_params(axis='x', rotation=25)  
            ax.set_ylim([0,1])

        plt.tight_layout()
        plt.show()

def plot_data_all_by_ue_categories(normalized_values_by_category):
    num_categories = len(normalized_values_by_category)
    spacing_factor = 1.5 
    bar_width = 0.15  

    #print("#### INSIDE PLOT PRINT ####")
    #print(normalized_values_by_category)

    fig, axes = plt.subplots(num_categories, 1, figsize=(10, 6*num_categories))

    for i, (category, values_dict) in enumerate(normalized_values_by_category.items()):
        ax = axes[i]

        ue_values = {k: v for k, v in values_dict.items() if k.startswith("ue")}
        ue_keys = list(ue_values.keys())
        ue_data = list(ue_values.values())
        num_ue = len(ue_keys)
        bar_positions = np.arange(len(ue_data[0]))

        colors = ['skyblue', 'lightgreen', 'lightcoral']  

        for j in range(num_ue):
            ax.bar(bar_positions + j * spacing_factor * bar_width, ue_data[j].values(), bar_width, label=f'{ue_keys[j]}', color=colors[j])

        presenting_values = values_dict.get("presenting_values", {})
        presenting_data = list(presenting_values.values())
        ax.bar(bar_positions + num_ue * spacing_factor * bar_width, presenting_data, bar_width, label='Presenting Values', color='orange')

        ax.set_title(f'Category: {category}')
        ax.set_ylabel('Normalized values')
        ax.set_xticks(bar_positions + num_ue * spacing_factor * bar_width / 2)
        ax.set_xticklabels(ue_data[0].keys())
        ax.tick_params(axis='x', rotation=25)
        ax.legend()

    plt.tight_layout()
    plt.show()



def plot_data_by_ue_and_metric(dict_to_plot):
    num_metrics = len(next(iter(dict_to_plot.values())))
    
    fig, axs = plt.subplots(num_metrics, figsize=(10, num_metrics*5))
    
    for i, metric in enumerate(next(iter(dict_to_plot.values()))):
        # Para cada categoria
        for j, category in enumerate(dict_to_plot):
            # Coleta os valores da métrica para cada UE
            values = list(dict_to_plot[category][metric].values())
            # Cria um gráfico de barras no subplot correspondente
            axs[i].bar(np.arange(len(values)) + j*0.3, values, width=0.3, label=category)
        
        # Configura o título e os rótulos do subplot
        axs[i].set_title(metric)
        axs[i].set_xticks(np.arange(len(values)))
        axs[i].set_xticklabels(list(next(iter(dict_to_plot[category].values())).keys()))
        axs[i].legend()
    
    plt.tight_layout()
    plt.savefig('pl_10_and_90.png', dpi=300)

    plt.show()

PRB_VALUES_TO_PLOT = [25, 52, 79, 106]
METRICS_TO_PLOT_PER_PRB = ['DRB.PacketSuccessRateUlgNBUu', 'DRB.UEThpUl', 'RRU.PrbAvailUl', 'RRU.PrbTotDl', 'RRU.PrbTotUl', 'DRB.RlcSduTransmittedVolumeUL', 'bitrate', 'jitter', 'lost_percentage', 'transfer', 'time_latency']
NOISE_AMPLITUDE_VALUES_TO_PLOT = [-28.0, -26.0, -24.0, -22.0, -20.0, -18.0, -17.8, -17.6]

def plot_metrics_av_per_prb(dict_to_plot):
    num_metrics = len(METRICS_TO_PLOT_PER_PRB)
    num_rows = num_metrics // 2 + (num_metrics % 2)  
    fig, axs = plt.subplots(num_rows, 2, figsize=(14, 7*num_rows))
    bar_width = 0.2

    colors = plt.cm.tab20c(np.linspace(0, 1, len(PRB_VALUES_TO_PLOT)))

    for i, metric in enumerate(METRICS_TO_PLOT_PER_PRB):
        values = [dict_to_plot[metric][prb] for prb in PRB_VALUES_TO_PLOT]
        r = list(range(len(PRB_VALUES_TO_PLOT)))

        col = i % 2
        row = i // 2

        axs[row, col].bar([x + bar_width for x in r], values, width=bar_width, edgecolor='white', color=colors)
        axs[row, col].set_xlabel('PRB', fontweight='bold')
        axs[row, col].set_xticks([r + bar_width for r in range(len(PRB_VALUES_TO_PLOT))], PRB_VALUES_TO_PLOT)
        axs[row, col].set_xticklabels(PRB_VALUES_TO_PLOT)
        axs[row, col].set_title(f'{metric} average')

    plt.tight_layout()

    plt.savefig('metrics_per_prb.png')
    
    
def plot_metrics_av_per_prb_and_an(dict_to_plot):
    num_metrics = len(METRICS_TO_PLOT_PER_PRB)
    fig, axs = plt.subplots(num_metrics, 1, figsize=(10, 6 * num_metrics))

    colors = plt.cm.tab20c(np.linspace(0, 1, len(PRB_VALUES_TO_PLOT)))

    for i, metric in enumerate(METRICS_TO_PLOT_PER_PRB):
        ax = axs[i]
        for j, prb in enumerate(PRB_VALUES_TO_PLOT):
            values = [dict_to_plot[metric][prb][an] for an in NOISE_AMPLITUDE_VALUES_TO_PLOT]
            ax.bar(np.arange(len(NOISE_AMPLITUDE_VALUES_TO_PLOT)) + j * 0.15, values, width=0.15, label=f'PRB {prb}', color=colors[j])

        ax.set_xlabel('Noise Amplitude Variation')
        ax.set_ylabel(metric)
        ax.set_title(f'PRB and An average of - {metric}')
        ax.set_xticks(np.arange(len(NOISE_AMPLITUDE_VALUES_TO_PLOT)) + 0.15 * (len(PRB_VALUES_TO_PLOT) - 1) / 2)
        ax.set_xticklabels(NOISE_AMPLITUDE_VALUES_TO_PLOT)
        ax.legend()

    plt.tight_layout()
    
    plt.savefig('metrics_per_prb_and_an.png')



BITRATE_VALUES_TO_PLOT = ['1M', '2M', '3M']
METRICS_TO_PLOT_PER_BITRATE = ['DRB.PacketSuccessRateUlgNBUu', 'DRB.UEThpUl', 'RRU.PrbAvailUl', 'RRU.PrbTotDl', 'RRU.PrbTotUl', 'DRB.RlcSduTransmittedVolumeUL', 'jitter', 'transfer', 'time_latency']
NOISE_AMPLITUDE_VALUES_TO_PLOT = [-28.0, -26.0, -24.0, -22.0, -20.0, -18.0, -17.8, -17.6, -17.4]

def plot_metrics_av_per_bitrate_and_an(dict_to_plot):
    num_metrics = len(METRICS_TO_PLOT_PER_BITRATE)
    fig, axs = plt.subplots(num_metrics, 1, figsize=(10, 6 * num_metrics))

    colors = plt.cm.tab20c(np.linspace(0, 1, len(BITRATE_VALUES_TO_PLOT)))

    for i, metric in enumerate(METRICS_TO_PLOT_PER_BITRATE):
        ax = axs[i]
        for j, bitrate in enumerate(BITRATE_VALUES_TO_PLOT):
            values = [dict_to_plot[metric][bitrate][an] for an in NOISE_AMPLITUDE_VALUES_TO_PLOT]
            ax.bar(np.arange(len(NOISE_AMPLITUDE_VALUES_TO_PLOT)) + j * 0.15, values, width=0.15, label=f'Bitrate {bitrate}', color=colors[j])

        ax.set_xlabel('Noise Amplitude Variation')
        ax.set_ylabel(metric)
        ax.set_title(f'PRB and An average of - {metric}')
        ax.set_xticks(np.arange(len(NOISE_AMPLITUDE_VALUES_TO_PLOT)) + 0.15 * (len(BITRATE_VALUES_TO_PLOT) - 1) / 2)
        ax.set_xticklabels(NOISE_AMPLITUDE_VALUES_TO_PLOT)
        ax.legend()

    plt.tight_layout()
    
    plt.savefig('metrics_per_bitrate_and_an.png')


BITRATE_VALUES_TO_PLOT = ['1M', '2M', '3M', '4M', '5M']
METRICS_TO_PLOT_PER_BITRATE = ['DRB.PacketSuccessRateUlgNBUu', 'DRB.UEThpUl', 'RRU.PrbAvailUl', 'RRU.PrbTotDl', 'RRU.PrbTotUl', 'DRB.RlcSduTransmittedVolumeUL', 'jitter', 'transfer', 'time_latency', 'bitrate']
NOISE_AMPLITUDE_VALUES_TO_PLOT = [-28.0, -26.0, -24.0, -22.0, -20.0, -18.0, -17.8, -17.6, -17.4]
PRB_VALUES_TO_PLOT = [52, 106]
OUTPUT_DIR = './latency_improved_plots'
"""def plot_metrics_av_per_bitrate_an_prb(metrics_dict):
    num_metrics = len(METRICS_TO_PLOT_PER_BITRATE)
    fig, axs = plt.subplots(num_metrics, 1, figsize=(15, 6 * num_metrics))

    # Verifica se há mais de um gráfico para plotar
    if num_metrics == 1:
        axs = [axs]

    colors = plt.cm.tab20c(np.linspace(0, 1, len(PRB_VALUES_TO_PLOT)))

    for i, metric in enumerate(METRICS_TO_PLOT_PER_BITRATE):
        ax = axs[i]
        for k, bitrate in enumerate(BITRATE_VALUES_TO_PLOT):
            for j, prb in enumerate(PRB_VALUES_TO_PLOT):
                values = [metrics_dict[metric][prb][bitrate][an] if metrics_dict[metric][prb][bitrate][an] is not None else 0 for an in NOISE_AMPLITUDE_VALUES_TO_PLOT]
                x_positions = np.arange(len(NOISE_AMPLITUDE_VALUES_TO_PLOT)) + j * 0.2 + k * 0.1
                ax.bar(x_positions, values, width=0.1, label=f'Bitrate {bitrate}, PRB {prb}', color=colors[j])

        ax.set_xlabel('Noise Amplitude')
        ax.set_ylabel(metric)
        ax.set_title(f'Average values for {metric}')
        ax.set_xticks(np.arange(len(NOISE_AMPLITUDE_VALUES_TO_PLOT)) + 0.1 * (len(BITRATE_VALUES_TO_PLOT) - 1) / 2)
        ax.set_xticklabels(NOISE_AMPLITUDE_VALUES_TO_PLOT)
        ax.legend()

    plt.tight_layout()
    plt.show()"""
def plot_metrics_av_per_bitrate_an_prb(metrics_dict):
    num_metrics = len(METRICS_TO_PLOT_PER_BITRATE)
    
    for metric in METRICS_TO_PLOT_PER_BITRATE:
        fig, axs = plt.subplots(len(BITRATE_VALUES_TO_PLOT), 1, figsize=(10, 6 * len(BITRATE_VALUES_TO_PLOT)), sharex=True)
        fig.suptitle(f'Average values for {metric}', fontsize=16)

        if len(BITRATE_VALUES_TO_PLOT) == 1:
            axs = [axs]

        colors = plt.cm.tab20c(np.linspace(0, 1, len(PRB_VALUES_TO_PLOT)))

        for k, bitrate in enumerate(BITRATE_VALUES_TO_PLOT):
            ax = axs[k]
            for j, prb in enumerate(PRB_VALUES_TO_PLOT):
                values = [metrics_dict[metric][prb][bitrate][an] if metrics_dict[metric][prb][bitrate][an] is not None else 0 for an in NOISE_AMPLITUDE_VALUES_TO_PLOT]
                x_positions = np.arange(len(NOISE_AMPLITUDE_VALUES_TO_PLOT)) + j * 0.2
                ax.bar(x_positions, values, width=0.2, label=f'PRB {prb}', color=colors[j])

            if k == len(BITRATE_VALUES_TO_PLOT) - 1:
                ax.set_xlabel('Noise Amplitude')
            ax.set_ylabel(metric)
            ax.set_title(f'Bitrate {bitrate}')
            ax.set_xticks(np.arange(len(NOISE_AMPLITUDE_VALUES_TO_PLOT)) + 0.1)
            ax.set_xticklabels(NOISE_AMPLITUDE_VALUES_TO_PLOT)
            ax.legend()

        output_path = os.path.join(OUTPUT_DIR, f'{metric}.jpg')
        plt.savefig(output_path)


OUTPUT_DIR = './latency_improved_plots/latency_only/'
def plot_latencies_per_test(latency_dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for test_number, ue_data in latency_dict.items():
        plt.figure(figsize=(120, 10)) 
        plt.title(f'Latencies for {test_number}', fontsize=16)

        for ue_key, latencies in ue_data.items():
            plt.plot(np.arange(len(latencies)), latencies, label=ue_key)

        plt.xlabel('Time')
        plt.ylabel('Time Latency (ms)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        output_path = os.path.join(OUTPUT_DIR, f'{test_number}.jpg')
        plt.savefig(output_path)
        plt.close()