import pandas as pd

left = pd.DataFrame({
    'key': ['K0', 'K1', 'K2', 'K3'],
    'A': ['A0', 'A1', 'A2', 'A3'],
    'B': ['B0', 'B1', 'B2', 'B3'],
})

right = pd.DataFrame({
    'key': ['K0', 'K1', 'K2', 'K3'],
    'C': ['C0', 'C1', 'C2', 'C3'],
    'D': ['D0', 'D1', 'D2', 'D3'],
})

print(left, end='\n---------------------\n')
print(right, end='\n---------------------\n')
print(pd.merge(left, right, on='key'), end='\n---------------------\n')

# how = ['inner', 'outer', 'left', 'right']
# inner
# pd.merge(left, right, on=['key1', 'key2'])
# pd.merge(left, right, on=['key1', 'key2'], how='outer')
# ...

# indicator：直观的显示
# pd.merge(left, right, on='key1', how='outer', indicator=True)

# index: let_index=True, right_index=True
# pd.merge(left, right, left_index=True, right_index=True)

boys = pd.DataFrame({
    'k': ['K0', 'K1', 'K2'],
    'age': [23, 24, 25]
})
girls = pd.DataFrame({
    'k': ['K0', 'K1', 'K2'],
    'age': [18, 19, 20]
})
print(pd.merge(boys, girls, on='k', suffixes=['_boy', '_girl']), end='\n---------------------\n')
