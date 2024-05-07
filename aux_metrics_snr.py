import os
import re
import matplotlib.pyplot as plt
import numpy as np


METRICS = ['DRB.RlcSduTransmittedVolumeUL', 'DRB.UEThpUl']

def get_metrics_from_files(dir):
    metrics_per_an = {}

    for filename in os.listdir(dir):
        if filename.startswith("an=") and filename.endswith(".txt"):
            # between = and .txt
            an_value = re.search(r'=(.*?).txt', filename).group(1)
            print(filename)
            print(an_value)
            metrics_per_an['an_'+an_value] = {metric: [] for metric in METRICS}

            filepath = os.path.join(dir, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if METRICS[0] in line:
                        metric_value = re.search(r'\[(.*?)\]', line).group(1)
                        try:
                            metrics_per_an['an_'+an_value][METRICS[0]].append(int(metric_value))
                        except: ### There are some values as None, they don't matter
                            pass
                    elif METRICS[1] in line:
                        metric_value = re.search(r'\[(.*?)\]', line).group(1)
                        try:
                            metrics_per_an['an_'+an_value][METRICS[1]].append(int(metric_value))
                        except:
                            pass

    return metrics_per_an

def mean_per_an(metric_dict_results):
    mean_metric_results = {}
    for key, value in metric_dict_results.items():
        mean_metric_results[key] = {}

        for metric in METRICS:
            if value[metric]:
                mean = round(sum(value[metric]) / len(value[metric]), 2)
                mean_metric_results[key][metric] = mean
            else:
                mean_metric_results[key][metric] = None

    return mean_metric_results

def plot_bar_chart(sorted_results):
    an_values = []
    rlc_sdu_transmitted_volume_means = []
    ue_thp_ul_means = []

    for key, value in sorted_results:
        if value['DRB.RlcSduTransmittedVolumeUL'] is not None and value['DRB.UEThpUl'] is not None:
            an_values.append(float(key.split('_')[1]))
            rlc_sdu_transmitted_volume_means.append(value['DRB.RlcSduTransmittedVolumeUL'])
            ue_thp_ul_means.append(value['DRB.UEThpUl'])

    sorted_indices = sorted(range(len(an_values)), key=lambda i: an_values[i])
    an_values = [an_values[i] for i in sorted_indices]
    rlc_sdu_transmitted_volume_means = [rlc_sdu_transmitted_volume_means[i] for i in sorted_indices]
    ue_thp_ul_means = [ue_thp_ul_means[i] for i in sorted_indices]

    bar_width = 0.35

    index = np.arange(len(an_values))

    fig, ax = plt.subplots()
    bars1 = ax.bar(index, rlc_sdu_transmitted_volume_means, bar_width, label='DRB.RlcSduTransmittedVolumeUL')
    bars2 = ax.bar(index + bar_width, ue_thp_ul_means, bar_width, label='DRB.UEThpUl')

    ax.set_xlabel('An value')
    ax.set_ylabel('Mean per UE')
    ax.set_title('Metrics mean by An value')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(an_values)
    ax.legend()

    plt.show()




def main():
    dir = "./snr_tests_files_v2"

    metric_dict_results = get_metrics_from_files(dir)

    sorted_mean_results = sorted(mean_per_an(metric_dict_results).items())

    for key, value in sorted_mean_results:
        print(key)
        for metric, mean in value.items():
            print(f"- {metric}: {mean}")

    plot_bar_chart(sorted_mean_results)
    


if __name__ == '__main__':
    main()


#for an, metricas in resultados.items():
#    print(f"Resultados para an={an}:")
#    for metrica, valores in metricas.items():
#        print(f"- {metrica}: {valores}")