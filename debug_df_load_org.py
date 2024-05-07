import pandas as pd

kpm_df_pl = pd.read_pickle("./pickles/srsran_kpms/df_kpms_new_org_simu_pl_10_f5.pkl")
kpm_pl_90 = pd.read_pickle("./pickles/srsran_kpms/df_kpms_new_org_simu_pl_90_f5.pkl")

iperf_df_pl = pd.read_pickle("./pickles/srsran_kpms/df_iperf_new_org_simu_pl_10_f5.pkl")
iperf_pl_90 = pd.read_pickle("./pickles/srsran_kpms/df_iperf_new_org_simu_pl_90_f5.pkl")

print("Columns df KPMs")
print(kpm_df_pl.columns)
print("Columns df iperf")
print(iperf_df_pl.columns)

#### Alteração so para fazer o join para que consiga ficar somente com o noise_amplitude na coluna dos kpms
iperf_df = iperf_df[['_time', 'noise_amplitude']]


### VERIFICAR SE É NECESSARIO!!!
### meter os dataframes nas colunas de tempo que quero dar join como objeto datetime para facilitar ações de join e garatnri que está tudo no mesmo formato
kpm_df['collectStartTime'] = pd.to_datetime(kpm_df['collectStartTime'])
iperf_df['_time'] = pd.to_datetime(iperf_df['_time'])

merged_df = pd.merge_asof(kpm_df, iperf_df, left_on='collectStartTime', right_on='_time', direction='nearest')

