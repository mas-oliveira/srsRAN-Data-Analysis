import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def correlation_matrix(df_corr):
    corr = df_corr.corr()
    #styled_corr = corr.style.background_gradient(cmap='coolwarm')
    #styled_corr.to_file('correlation_matrix.html')
    ax = sns.heatmap(corr, annot=True)
    plt.show()

