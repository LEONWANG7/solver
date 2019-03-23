import pandas as pd
import numpy as np

s = pd.Series([1, 3, 4, np.nan, 43, 1])
print(s, end='\n---------------------\n')

dates = pd.date_range('20160101', periods=6)
print(dates, end='\n---------------------\n')

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['a', 'b', 'c', 'd'])
print(df, end='\n---------------------\n')

df2 = pd.DataFrame({
    'A': [1., 2., 3., 4.],
    'B': pd.Timestamp('20130102'),
    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
    'D': np.array([3] * 4, dtype='int32'),
    'E': pd.Categorical(['test', 'train', 'test', 'train'], ),
    'F': 'foo'
})
print(df2, end='\n---------------------\n')
print(df2.dtypes, end='\n---------------------\n')
print(df2.index, end='\n---------------------\n')
print(df2.columns, end='\n---------------------\n')
print(df2.values, end='\n---------------------\n')
print(df2.describe(), end='\n---------------------\n')
print(df2.T, end='\n---------------------\n')
print(df2.sort_index(axis=1, ascending=False), end='\n---------------------\n')
print(df2.sort_values(by='E'), end='\n---------------------\n')


