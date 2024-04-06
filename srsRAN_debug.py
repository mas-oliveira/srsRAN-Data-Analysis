def write_csv(df, name):
    df.to_csv(f'./helpers/debug_df/{name}.csv')