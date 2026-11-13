import pandas as pd


df = pd.read_csv('td_report.csv', delimiter='\t')
print( df[-2:-1][['Prec', 'Rec', 'F1']])


df = pd.read_csv('toc_report.csv', delimiter='\t')
tbl = df[-4:-3]
print(tbl[['Inex08-P', 'Inex08-R', 'Inex08-F1', 'Inex08-Title acc', 'Inex08-Level acc']])

