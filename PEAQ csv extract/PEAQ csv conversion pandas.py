import sys
from io import StringIO
import pandas as pd
import glob

nomes_arq = glob.glob('*.csv')
dataframes = []
colunas =  ['Serie', 'DQ', 'Vol', 'Xt', 'Mt', 'XMT', 'DH', 'Fit', 'Residual', 'Extra']

for nome in nomes_arq:
    raw = open(nome, 'r', encoding = 'utf8').read().replace('","',';').replace(',"',';').replace('"\n','\n').replace('e0",','e0;')
    temp = StringIO(raw)
    df = pd.read_csv(temp, sep=';', decimal=',', names=colunas, header=0)
    dataframes.append(df)

#print(dataframes)
conjunto = pd.DataFrame()

for i, dataframe in enumerate(dataframes):
    Xt = dataframe['Xt'].copy()[1:].reset_index(drop=True)
    DH = dataframe['DH'].copy()[:-1].reset_index(drop=True)
    juntos = pd.concat([Xt, DH], axis=1)
    juntos.columns = ['Xt_' + nomes_arq[i][:-4], 'DH_' + nomes_arq[i][:-4]]
    conjunto = pd.concat([conjunto, juntos], axis=1)

    
#print(conjunto)
conjunto.to_csv('final.dat', sep=';', decimal=',', index=False)