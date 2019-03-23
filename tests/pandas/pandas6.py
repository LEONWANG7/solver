import pandas as pd
import numpy as np

df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['A', 'B', 'C', 'D'])
df2 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['A', 'B', 'C', 'D'])
df3 = pd.DataFrame(np.ones((3, 4)) * 2, columns=['A', 'B', 'C', 'D'])
print(df1, end='\n---------------------\n')
print(df2, end='\n---------------------\n')
print(df3, end='\n---------------------\n')
df = pd.concat([df1, df2, df3], axis=0)
print(df, end='\n---------------------\n')
df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
print(df, end='\n---------------------\n')

# join={'inner', 'outer'}
df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['A', 'B', 'C', 'D'], index=[1, 2, 3])
df2 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['B', 'C', 'D', 'E'], index=[2, 3, 4])
print(df1, end='\n---------------------\n')
print(df2, end='\n---------------------\n')
# 外连接
print(pd.concat([df1, df2]), end='\n---------------------\n')
# 内连接
print(pd.concat([df1, df2], join='inner'), end='\n---------------------\n')
# 左右连接
print(pd.concat([df1, df2], axis=1), end='\n---------------------\n')
print(pd.concat([df1, df2], axis=1, join_axes=[df1.index]), end='\n---------------------\n')
# append
print(df1.append(df2, ignore_index=True), end='\n---------------------\n')
s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
print(df1.append(s1, ignore_index=True), end='\n---------------------\n')

