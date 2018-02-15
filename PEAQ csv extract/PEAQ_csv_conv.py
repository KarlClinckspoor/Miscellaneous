import glob
import locale

loc = locale.setlocale(locale.LC_ALL, '')
loc_settings = locale.localeconv()
sys_sep = loc_settings['decimal_point']

#print('sys_sep ' + sys_sep)

nomes_arq = glob.glob('*.csv')
amostras = []
comprimentos = []

#colunas =  ['Serie', 'DQ', 'Vol', 'Xt', 'Mt', 'XMT', 'DH', 'Fit', 'Residual', 'Extra']

for nome in nomes_arq:
    print('Extraindo informações do arquivo', nome, end = '')
    Xt = []
    DH = []
    raw = open(nome, 'r', encoding = 'utf8').read().split('\n')
    
    if len(raw[0].split(';')) > 2:
        item_sep = ';'
        linhas = [item.replace('","',';').replace(',"',';').replace('"\n','\n').replace('e0",','e0;') for item in raw]
    elif len(raw[0].split(',')) > 2:
        item_sep = ','
        linhas = raw
    
    #print('file sep' + sep)
    
    # if loc_settings['decimal_point'] == ',':
        # raw = open(nome, 'r', encoding = 'utf8').read().replace('","',';').replace(',"',';').replace('"\n','\n').replace('e0",','e0;')
    # elif loc_settings['decimal_point'] == '.':
        # raw = open(nome, 'r', encoding = 'utf8').read()
    # else:
        # print('Unknown decimal separator')
        
    #linhas = raw.split('\n')
    for linha in linhas:
        if len(linha) <= 5:
            continue
        linha = linha.split(item_sep)
        if len(linha[1].split('.')) == 2:
            dec_sep = '.'
        elif len(linha[1].split(',')) == 2:
            dec_sep = ','
        dec_sep = '.' 
        Xt.append(linha[3].replace(dec_sep, sys_sep))
        DH.append(linha[6].replace(dec_sep,sys_sep))
    Xt[0] = nome + ' ' + Xt[0]
    DH[0] = nome + ' ' + DH[0]
    del Xt[1]
    del DH[-1]
    amostras.append(Xt)
    amostras.append(DH)
    comprimentos.append(len(Xt))
    print(' Pronto.')
    

conteudo = list(range(0,max(comprimentos)))

for i in range(0,max(comprimentos)):
    #conteudo[i] = [amostra[i] for amostra in amostras]#funciona, não aceita diferentes tamanhos
    a_adicionar = []
    for amostra in amostras:
        try:
            a_adicionar.append(amostra[i])
        except IndexError:
            a_adicionar.append('')
    conteudo[i] = a_adicionar

with open('dest.dat','w',encoding='utf8') as fhand:
    for i in range(len(conteudo)-1):
        fhand.write(';'.join(conteudo[i]))
        fhand.write('\n')
