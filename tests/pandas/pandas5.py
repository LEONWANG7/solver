import pandas as pd

data = pd.read_csv('student.csv')
print(data, end='\n---------------------\n')

data.to_pickle('student.pickle')
