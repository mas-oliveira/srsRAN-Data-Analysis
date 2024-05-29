import pandas as pd

import srsRAN_debug

df_kpm = pd.read_pickle('./pickles/srsran_kpms/prbs/df_kpms_prbs_agg.pkl')
df_iperf = pd.read_pickle('./pickles/srsran_kpms/prbs/df_iperf_prbs_agg.pkl')
df_latency = pd.read_pickle('./pickles/srsran_kpms/prbs/df_latency_prbs_agg.pkl')

print(df_kpm.columns)
print(df_iperf.columns)
print(df_latency.columns)
df_kpm.rename(columns={'_time_x': '_time'}, inplace=True)

df_kpm.rename(columns={'ue_nr': 'ue_nr_kpm'}, inplace=True)
df_iperf.rename(columns={'ue_nr': 'ue_nr_iperf'}, inplace=True)
df_latency.rename(columns={'ue_id': 'ue_nr_latency'}, inplace=True)

df_kpm['_time'] = pd.to_datetime(df_kpm['_time']).dt.tz_localize(None)
df_iperf['_time'] = pd.to_datetime(df_iperf['_time']).dt.tz_localize(None)
df_latency['_time'] = pd.to_datetime(df_latency['_time']).dt.tz_localize(None)

df_result = pd.merge(df_kpm, df_iperf, on='_time', how='outer')
df_result = pd.merge(df_result, df_latency, on='_time', how='outer')

df_result['noise_amplitude'] = df_result['noise_amplitude_x'].combine_first(df_result['noise_amplitude_y'])
df_result['prb'] = df_result['prb_x'].combine_first(df_result['prb_y'])

df_result.drop(columns=['noise_amplitude_x', 'noise_amplitude_y', 'prb_x', 'prb_y'], inplace=True)

srsRAN_debug.write_csv(df_result, 'debug_merge')
df_result.to_pickle('./pickles/srsran_kpms/prbs/completely_agg_prb_an.pkl')

######## Para agregar sem perder tanta informação podes pegar no join que já fizeste.
### Passo 1- Separar por PRB
### Passo 2- Separar por test_number
### Passo 3- Fazer a agregação das linhas até haver uma com o _time comum e com os 3 sets de KPMs   
###                existentes. Quando ocorrer esta situação então a forma de agregar vai ser ao
###                calcular a média das prévias ocorrências desse KPM em questão.