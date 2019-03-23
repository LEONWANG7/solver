import pandas as pd
import numpy as np

dates = pd.date_range('20190101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
print(df, end='\n---------------------\n')
print(df['A'], end='\n---------------------\n')
print(df.A, end='\n---------------------\n')
print(df[0:3], end='\n---------------------\n')
print(df['20190102':'20190106'], end='\n---------------------\n')
# 根据标签选择
print(df.loc['20190102'], end='\n---------------------\n')
print(df.loc[:, ['A', 'B']], end='\n---------------------\n')
print(df.loc['20190102', ['A', 'B']], end='\n---------------------\n')
# 根据位置选择
print(df.iloc[3], end='\n---------------------\n')
print(df.iloc[3:5, 1:3], end='\n---------------------\n')
print(df.iloc[[1, 3, 5], 1:3], end='\n---------------------\n')
# 综合选择
print(df.ix[:3, ['A', 'C']], end='\n---------------------\n')
# 布尔筛选
print(df[df.A > 8], end='\n---------------------\n')
