from sklearn.preprocessing import minmax_scale
import pandas as pd
import matplotlib.pyplot as plt
import srsRAN_data_treatment
import numpy as np

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