import matplotlib.pyplot as plt
import pandas as pd
#import scipy
#import statsmodels.api as sm
import glob
from lmfit import minimize, Parameters, Parameter, report_fit
import numpy as np
from scipy.optimize import curve_fit

"""
Works well with Spyder, because it shows the graps inline. To change between versions, find where it says
Spyder and edit the code accordingly.
"""
# todo: escrever no arquivo de resultados o modelo de ajuste utilizado.
# todo: plotar curvas mostrando o ajuste, com barra de erro
# todo: no modo automático, salvar as figuras com o dado e o plot, para ver depois.
# todo: no modo normal, plotar as figuras e dps perguntar se quer salvar elas.
# todo: tentar ser inteligente e encontrar a coluna com GP e Eta


def fit_lin_min(params, x, y):
    # For lmfit
    a = params['a'].value
    b = params['b'].value

    model = a + b * x
    return model - y

    
def fit_lin(x, a, b):
    return a + b * x


def ExtractData(fname, FC_segment=0):
    fhand = open(fname, 'r')
    GP = []
    Eta = []
    column_eta = 0
    column_gp = 0
    #FC_segment = '3'
    
    #while FC_segment == 0:
    #    FC_segment = input("What is the segment that has the flow curves? (eg. [1], 2, 3) If you do not know, don't write anything. ")
    #    if FC_segment == '':
    #        print(fhand.read())
    #    elif FC_segment.isnumeric():
    #        break
    #    else:
    #        print('Not a valid number')
    
    for line in fhand:
        if line.startswith(';'):
            column_names = line.rstrip().split(';')
            #print('Debug: column names', column_names)
            for i, column in enumerate(column_names):
                if 'Eta' in column:
                    column_eta = i
                    #print('Debug: Found Eta at', column_eta)
                if 'GP' in column:
                    column_gp = i
                    #print('Debug: Found GP at', column_gp)
        #print('Debug: split line', line.split(';'))
        #print('Debug: is numeric', line.replace(',','.').split(';')[column_eta], line.replace(',','.').split(';')[column_eta].isnumeric())
        try:
            GP.append(float(line.replace(',','.').split(';')[column_gp]))
            Eta.append(float(line.replace(',','.').split(';')[column_eta]))
            #print('Debug: GP: ', GP, 'Eta', Eta)
            #if line.replace(',','.').split(';')[column_eta].isnumeric():
            #    num, gp, tau, eta, *rest = line.replace(',','.').split(';')
                
                #GP.append(float(gp))
                #Eta.append(float(eta))
                
        except:
            pass
        
        #if line.startswith(FC_segment + '|'):
        #    line = line.rstrip()
        #    num, gp, tau, eta, *rest = line.replace(',','.').split(';')
        #    GP.append(float(gp))
        #    Eta.append(float(eta))
        #    #print(line)
    
    fhand.close()
    if len(GP) == 0:
        print('No Flow Curve data was found! Re-export the data')
        quit()
    #return pd.Series(GP), pd.Series(Eta)
    return GP, Eta

def PlotData(GP, Eta):
    #labels = [i in range(0, len(GP), 1)]
    plt.xscale('log')
    plt.yscale('log')

    #plt.ion()

    plt.scatter(GP, Eta, marker='*')
    for i in range(0, len(GP)):
        plt.annotate(str(i), (GP[i], Eta[i]))
    
    #Other: uncomment the following 2 lines.
    #plt.draw()
    #plt.pause(0.001)
    
    #Spyder: uncomment the following line.
    plt.show()

    return None

def Average_mean(Eta):
    #while True:
    #    try:
    #        initial = int(input('What is the first point you want to use? '))
    #        final = int(input('What is the final point you want to use? '))
    #        break
    #    except ValueError:
    #        print('Not a valid number')
    #        continue

    #x_int = pd.Series(GP[initial:final])
    #y_int = Eta[initial:final]
    aver = pd.Series(Eta).mean()
    aver_err = pd.Series(Eta).std()
    return aver, aver_err
  

def record(name, aver, aver_err, silent = False, extra = ''):
    if silent == False:
        print(name + ':', 'Intercept', aver, '+-', aver_err)

    with open('results.dat', 'a', encoding = 'utf-8') as fdest:
        fdest.write(name + ';' + str(aver) + ';' + str(aver_err) + extra +'\n')

        
def TreatFile(file, model='fitting'):
    GP, Eta = ExtractData(file)

    PlotData(GP, Eta)

    while True:
        try:
            initialp = int(input('What is the first point you want to use? '))
            finalp = int(input('What is the final point you want to use? '))
            break
        except ValueError:
            print('Not a valid number')
            continue

    #model = 'fitting'
    if model == 'lin':
        print('Attempting to linearize the points')
        GP_arr = np.array(GP[initialp:finalp+1])
        Eta_arr = np.array(Eta[initialp:finalp+1])
        params = Parameters()
        # model: a + b * x
        params.add('a', value = 0, min = 0.0, max = 100)
        params.add('b', value = 0, min = 0.0, max = 10)

        result = minimize(fit_lin_min, params, args=(GP_arr, Eta_arr))
        final = Eta_arr + result.residual

        plt.plot(GP_arr, final)
        #Other: uncomment the following 2 lines:
        #plt.draw()
        #plt.pause(0.001)
        
        #Spyder: uncomment the following line
        plt.show()
        

        report_fit(result)
        a = result.params['a'].value
        aerr = result.params['a'].stderr

        record(file, a, aerr)
        #return

    if model == 'mean':
        print('Averaging the points')
        aver, aver_err = Average_mean(Eta[initialp:finalp+1])
        record(file, aver, aver_err, extra=str(initialp)+str(finalp+1))
        #return
    
    if model == 'fitting':
        GP_arr = np.array(GP[initialp:finalp+1])
        Eta_arr = np.array(Eta[initialp:finalp+1])
        popt, pcov = curve_fit(fit_lin, GP_arr, Eta_arr, p0=(30, 0), 
                               bounds=(0, [1000., 0.0001])) #
        perr = np.sqrt(np.diag(pcov))

        a = popt[0]
        aerr = perr[0]
        record(file, a, aerr, extra='first' + str(initialp) + 'final' + 
               str(finalp+1))
        #return
    do_plot = 'Do you want to plot the data? y/[n]'
    if do_plot == 'y':
        plot_fitted_data(file, GP, Eta, a, aerr)

    
# todo: Salvar nos plots os valores de a, aerr, e os pontos usados para fazer os fittings.
def plot_fitted_data(file, GP, Eta, a, aerr, do_save=True):
    #print('Debug: GP', GP, 'Eta', Eta)
    x = np.logspace(np.log10(GP[0]), np.log10(GP[-1]))
    #print('Debug: x', x)
    y = np.ones(len(x)) * a
    #print('Debug: y', y)
    yerr = np.ones(len(x)) * aerr
    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(GP, Eta)
    plt.errorbar(x, y, yerr = yerr)
    if do_save:
        plt.savefig(file[:-4] + '.png')
        print('Figure saved.')
    plt.show()
    return


def automaticFitting(file):
    fittings = []
    #segment = input('What is the segment that contains the flow curves for all the data?')
    GP, Eta = ExtractData(file)  #, FC_segment = segment)
    length = len(GP)
    
    for first_point in range(0, length // 3, 1):
        for last_point in range(first_point + 3, length // 2, 1):
            GP_arr = np.array(GP[first_point:last_point + 1])
            Eta_arr = np.array(Eta[first_point:last_point + 1])
            popt, pcov = curve_fit(fit_lin, GP_arr, Eta_arr, p0=(30, 0), 
                                   bounds=(0, [1000., 0.0001]))
            perr = np.sqrt(np.diag(pcov))
            fittings.append((first_point, last_point, popt, perr))
    
    #print('Debug: fittings: ', fittings)
    fittings.sort(key = lambda x: x[3][0])
    a = fittings[0][2][0]
    aerr = fittings[0][3][0]
    #print('Debug: fittings_sorted: ', fittings)
    #print('Debug: a: ', a)
    #print('Debug: aerr: ', aerr)
    record(file, a, aerr, extra = ' first ' + str(fittings[0][0]) + ' last ' +
           str(fittings[0][1]))
    #print('Debug: AutoFitting: GP', GP, 'Eta', Eta)
    plot_fitted_data(file, GP, Eta, a, aerr)
    return
            

#GP_pd = pd.Series(data = GP)
#Eta_pd = pd.Series(data = Eta)
# scipy.optimize.curve_fit(lambda x, m: m*x, xi, y)

#model = sm.OLS(Y, X_2)
#results = model.fit()
#par = results.params

if __name__ == '__main__':
    all_or_select = input('Do you want to treat (a)ll files in the folder or a few (s)pecific files?')
    automatic_or_manual = input('Do you want to (s)elect the points or treat them (a)utomatically?')
    
    if all_or_select.lower() == 'a' or all_or_select.lower() == 'all': # todo: mudar para all_or_select in valid_choices
        while True:
            ext = input("What is the file extension? Be careful, only select the relevant files?")
            allfilenames = glob.glob('*.' + ext)
            if len(allfilenames) > 0:
                break
            else:
                print('File not found')
        if automatic_or_manual == 's':
            for file in allfilenames:
                TreatFile(file)
        elif automatic_or_manual == 'a':
            for file in allfilenames:
                automaticFitting(file)
    
    else:
        while True:
            ext = input('What is the file extension?')
            if ext == 'exit':
                break
            files = glob.glob('*.{}'.format(ext.lower()))
            if len(files) == 0:
                print('No file found with that extension')
                continue
            # pretty printer?
            for index, file in enumerate(files):
                print(index, file)
            while True:
                try:
                    selected_file = files[int(input('Which file number?'))]
                    break
                except ValueError:
                    print('Not a number')
                except IndexError:
                    print('This file number is not present')
        
            if automatic_or_manual == 's':
                TreatFile(selected_file)
            elif automatic_or_manual == 'a':
                automaticFitting(selected_file)
            
            plt.clf()
            if input('Do you want to treat another file? y/[n]') == 'y':
                continue
            else:
                plt.close()
                break