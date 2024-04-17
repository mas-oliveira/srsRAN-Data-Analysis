import pandas as pd

# Lê o arquivo CSV em um DataFrame
df = pd.read_csv('kpms.csv')

# Converte a coluna '_time' para datetime
df['_time'] = pd.to_datetime(df['_time'])

# Define os horários de início e fim
start_time = pd.to_datetime('2024-04-09 15:37:41+00:00')
end_time = pd.to_datetime('2024-04-09 16:37:38+00:00')

# Filtra o DataFrame para incluir apenas as linhas dentro do intervalo de tempo especificado
filtered_df = df[(df['_time'] >= start_time) & (df['_time'] <= end_time)]

# Calcula a média de cada coluna no DataFrame filtrado
mean_values = filtered_df.mean()

# Imprime os valores médios
print(mean_values)
