import pandas as pd
import numpy as np

dates = pd.date_range('20190101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
print(df, end='\n---------------------\n')
df.iloc[2, 2] = 666
df.loc['20190101', 'B'] = 111
df.A[df.A < 8] = np.nan
df['F'] = np.nan
df['E'] = [1, 2, 3, 4, 5, 6]
print(df, end='\n---------------------\n')
