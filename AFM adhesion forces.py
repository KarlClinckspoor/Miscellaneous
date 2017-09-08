#%%
import matplotlib.pyplot as plt
import numpy as np
import glob
#%%
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]

#%%
def trim_files():
    names = glob.glob('*.txt')
    for name in names:
        fhand = open(name,'r')
        dest = name[:-4]+'_int.txt'
        fdest = open(dest,'w')
        
        for line in fhand:
            linha = line.split('\t')
            if len(linha) < 3:
                continue
            x = linha[3]
            y = linha[2]
            dado = x + ' ' + y + '\n'
            fdest.write(dado)
        
        fdest.close()
        fhand.close()
#%%
def find_deltay(fname):
    dados = np.loadtxt(fname,delimiter=' ', skiprows=2)
    dados_x = dados[:,0]
    dados_y = dados[:,1]
    
    plt.plot(dados_x,dados_y)
    plt.title(fname)
    name = fname+'.png'
    plt.savefig(name,dpi=150)
    #plt.show()
    plt.clf()   
    
    minimo_y_val = min(dados_y)
    minimo_y_pos = dados_y.argmin()
    #minimo_x_val = dados_x[minimo_y_pos]
    #minimo_x_pos = minimo_y_pos
    
    
    interesse = dados_y[minimo_y_pos:]
    interesse_x = dados_x[minimo_y_pos:]
    '''
    derivada_interesse = np.diff(interesse, n=1)
    min_derivada = min(derivada_interesse)
    min_derivada_pos = derivada_interesse.argmin()
    
    inflex_y_val = interesse[min_derivada_pos]
    '''
    interesse_y_inflex_pos = 0
    interesse_y_inflex_value = 0
    
    for index, value in enumerate(interesse):
        coef, residuals, rank,singular,rcond = np.polyfit(interesse_x[index:], interesse[index:], 1, full=True)
        if len(residuals) == 0: continue
        if residuals[0]<1:
            interesse_y_inflex_pos = index
            interesse_y_inflex_value = value
            break
        #print(index, residuals)
    
    #inflex_y_val = interesse[min_derivada_pos]
    
    #delta_y = inflex_y_val - minimo_y_val
    #print(delta_y)
    
    delta_y_2 = interesse_y_inflex_value - minimo_y_val
    return delta_y_2
#delta_y = min_interesse - minimo_y_val

#interesse = dados_y[minimo_y_pos:]

#%%
if __name__ == '__main__':
    trim_files()
    files = glob.glob('*int.txt')
    fhand = open('resultados.txt','w')
    for file in files:
        delta_y = find_deltay(file)
        text = file+' '+str(delta_y)+'\n'
        fhand.write(text)
    fhand.close()
    
#%%
'''
derivada = list()
for index, value in enumerate(interesse):
    dif_y = interesse[index+1]-interesse[index]
    derivada.append(dif_y)
    maximo_y = max(derivada)
    maximo_x = interesse[]
'''
#%%

'''
fhand = open('dadoliadest.txt','r')
x = list()
y = list()
for line in fhand:
   
    if count == 0:
        continue
    if count == 1:
        continue
    count += 1
 
    dado = line.split(' ')
    try:
        dado_x= float(dado[0])
        dado_y= float(dado[1])
    except:
        dado_x = 0
        dado_y = 0
    x.append(dado_x)
    y.append(dado_y)

#print(x)
#print(y)

y_min = min(y)
y_max = max(y[500:])

print(y_min)
print(y_max)

plt.plot(x,y)

plt.show()
fhand.close()
'''
#%%

#%%
'''
dados = np.loadtxt('dadoliadest.txt',delimiter=' ')
dados_x = dados[:,0]
dados_y = dados[:,1]
#plt(dados_x,dados_y)
minimo_y_val = min(dados_y)
minimo_y_pos = dados_y.argmin()
minimo_x_val = dados_x[minimo_y_pos]
minimo_x_pos = minimo_y_pos


interesse = dados_y[minimo_y_pos:]
interesse_x = dados_x[minimo_y_pos:]
'''
'''
derivada_interesse = np.diff(interesse, n=1)
min_derivada = min(derivada_interesse)
min_derivada_pos = derivada_interesse.argmin()

inflex_y_val = interesse[min_derivada_pos]

interesse_y_inflex_pos = 0
interesse_y_inflex_value = 0
'''
'''
for index, value in enumerate(interesse):
    coef, residuals, rank,singular,rcond = np.polyfit(interesse_x[index:], interesse[index:], 1, full=True)
    if len(residuals) == 0: continue
    if residuals[0]<1:
        interesse_y_inflex_pos = index
        interesse_y_inflex_value = value
        break
    #print(index, residuals)

#inflex_y_val = interesse[min_derivada_pos]

delta_y = inflex_y_val - minimo_y_val
print(delta_y)

delta_y_2 = interesse_y_inflex_value - minimo_y_val
'''