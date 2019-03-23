import pandas as pd
import numpy as np

dates = pd.date_range('20190101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
df.iloc[0, 1] = np.nan
df.iloc[4, 2] = np.nan
print(df, end='\n---------------------\n')
print(df.dropna(axis=0, how='any'), end='\n---------------------\n')  # how={'any', 'all'}

dates = pd.date_range('20190101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
df.iloc[0, 1] = np.nan
df.iloc[4, 2] = np.nan
print(df.fillna(value=-1), end='\n---------------------\n')  # how={'any', 'all'}

dates = pd.date_range('20190101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
df.iloc[0, 1] = np.nan
df.iloc[4, 2] = np.nan
print(np.any(df.isna()) == True, end='\n---------------------\n')  # how={'any', 'all'}

print(df.isna().sum().sum(), end='\n---------------------\n')  # how={'any', 'all'}
