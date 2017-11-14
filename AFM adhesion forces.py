# encoding = 'utf-8'
# Written for my friend Lia, to save her a few hours, and her insanity, from having to treat the AFM data.

import matplotlib.pyplot as plt
import numpy as np
import glob


def rewrite_files():
    """Finds all the .txt files in the directory and rewrites them in a more straightforward way for what is \
    interesting in this script."""
    names = glob.glob('*.txt')
    for name in names:
        with open(name, 'r') as fsource:
            with open((name[:-4] + '_int.txt'), 'w') as fdest:
                for line in fsource:
                    line = line.split('\t')
                    if len(line) < 3:
                        continue
                    x = line[3]
                    y = line[2]
                    interest = x + ' ' + y + '\n'
                    fdest.write(interest)


#
def trim_files():
    """Deprecated. The function rewrite_files is better."""
    names = glob.glob('*.txt')
    for name in names:
        fhand = open(name, 'r')
        dest = name[:-4] + '_int.txt'
        fdest = open(dest, 'w')
        
        for line in fhand:
            line = line.split('\t')
            if len(line) < 3:
                continue
            x = line[3]
            y = line[2]
            dado = x + ' ' + y + '\n'
            fdest.write(dado)
        
        fdest.close()
        fhand.close()


# todo: rewrite this to use pandas dataframes if better.
def find_delta_y(fname):
    """Finds the delta Y, or the adhesion force, by finding the minimum of the dataset and then finding the \
    first point where the curve becomes linear."""
    dados = np.loadtxt(fname, delimiter=' ', skiprows=2)  # requires a properly formatted file.
    dados_x = dados[:, 0]
    dados_y = dados[:, 1]
    
    plt.plot(dados_x, dados_y)
    plt.title(fname)
    name = fname + '.png'
    plt.savefig(name, dpi=150)
    plt.clf()   
    
    minimo_y_val = min(dados_y)
    minimo_y_pos = dados_y.argmin()
    # minimo_x_val = dados_x[minimo_y_pos]
    # minimo_x_pos = minimo_y_pos

    # The region of interest if right after the minimum of the curve.
    interest = dados_y[minimo_y_pos:]
    interest_x = dados_x[minimo_y_pos:]
    '''
    derivada_interesse = np.diff(interest, n=1)
    min_derivada = min(derivada_interesse)
    min_derivada_pos = derivada_interesse.argmin()
    
    inflex_y_val = interest[min_derivada_pos]
    '''
    interesse_y_inflex_pos = 0
    interesse_y_inflex_value = 0

    # Now that the minimum is found, go through all the remaining points, starting at the minimum and going up,
    # and fit them linearly. As soon as the fit becomes good (residuals < 1), stops and returns that point.
    # [-------] -> [X------] -> [XX-----], where X are excluded points.
    for index, value in enumerate(interest):
        coef, residuals, rank, singular, rcond = np.polyfit(interest_x[index:], interest[index:], 1, full=True)
        # Only residuals is of interest.
        if len(residuals) == 0:
            continue
        if residuals[0] < 1:
            interesse_y_inflex_pos = index
            interesse_y_inflex_value = value
            break
        # print(index, residuals)
    
    delta_y = interesse_y_inflex_value - minimo_y_val
    return delta_y


# %%
if __name__ == '__main__':
    rewrite_files()
    files = glob.glob('*int.txt')
    with open('resultados.txt', 'w') as fhand:
        for file in files:
            delta_y = find_delta_y(file)
            text = file+' '+str(delta_y)+'\n'
            fhand.write(text)
