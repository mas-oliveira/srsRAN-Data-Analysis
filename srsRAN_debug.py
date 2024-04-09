import pandas as pd

def write_csv(df, name):
    df.to_csv(f'./helpers/debug_df/{name}.csv')

def write_list_csv(df_list, name):
    # Concatenate all dataframes in the list
    combined_df = pd.concat(df_list)
    # Write the combined dataframe to a CSV file
    combined_df.to_csv(f'./helpers/debug_df/{name}.csv')
