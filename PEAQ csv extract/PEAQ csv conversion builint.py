import glob

nomes_arq = glob.glob('*.csv')
amostras = []
comprimentos = []

#colunas =  ['Serie', 'DQ', 'Vol', 'Xt', 'Mt', 'XMT', 'DH', 'Fit', 'Residual', 'Extra']

for nome in nomes_arq:
    Xt = []
    DH = []
    raw = open(nome, 'r', encoding = 'utf8').read().replace('","',';').replace(',"',';').replace('"\n','\n').replace('e0",','e0;')
    linhas = raw.split('\n')
    for linha in linhas:
        if len(linha) <= 5:
            continue
        linha = linha.split(';')
        Xt.append(linha[3])
        DH.append(linha[6])
    Xt[0] = nome + Xt[0]
    DH[0] = nome + DH[0]
    del Xt[1]
    del DH[-1]
    amostras.append(Xt)
    amostras.append(DH)
    comprimentos.append(len(Xt))

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
