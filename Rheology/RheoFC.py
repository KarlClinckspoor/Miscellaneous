import matplotlib.pyplot as plt
import pandas as pd
#import scipy
#import statsmodels.api as sm
import glob
#from lmfit import minimize, Parameters, Parameter, report_fit
import numpy as np
from scipy.optimize import curve_fit

"""
Works well with Spyder, because it shows the graps inline. To change between versions, find where it says
Spyder and edit the code accordingly.
"""

#######
# Settings:
# Shows additional messages to help in debugging.
DEBUG = False
# If what you're running Python on supports inline graphs, like iPython in Spyder does, set this to True. 
# Otherwise, set this to False, so that you can use the program while the Graph is open.
SPYDER = False
# Change the sorting method to either 'by_error' or 'by_error_length'
SORTING = 'by_error'
########

def help():
    help_file = open('help', 'r', encoding = 'utf-8').read()
    print(help_file)

def fit_lin_min(params, x, y):
    """Function used for the lmfit part. Largely unnecessary and unreliable."""
    a = params['a'].value
    b = params['b'].value

    model = a + b * x
    return model - y

    
def fit_lin(x, a, b):
    """Simple function for a linear fit, with a as the linear coefficient and b the angular coefficient."""
    return a + b * x


def ExtractData(fname, FC_segment=0):
    """Opens the file fname and extracts the data based on where it finds the word 'Eta' and 'GP', these being
    the Viscosity and the Shear Rate (gamma point). If the file has multiple segments, for example, when multiple
    experiments were done in succession, FC_segment indicates which of those experiments was a Flow Curve."""
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
            if DEBUG:
                print('Debug: column names', column_names)
            for i, column in enumerate(column_names):
                if 'Eta' in column and 'Eta*' not in column:
                    column_eta = i
                    if DEBUG:
                        print('Debug: Found Eta at', column_eta)
                if 'GP' in column:
                    column_gp = i
                    if DEBUG:
                        print('Debug: Found GP at', column_gp)
        try:
            GP.append(float(line.replace(',','.').split(';')[column_gp]))
            Eta.append(float(line.replace(',','.').split(';')[column_eta]))
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
    if DEBUG:
        print('Debug: Extracted Data: GP:', GP, 'Eta:', Eta)
    return GP, Eta

def PlotData(GP, Eta):
    """Plots the data with labels accompanying each point, for easier attribution.
    Depending on where the program is being run, it is better to leave """
    #labels = [i in range(0, len(GP), 1)]
    plt.xscale('log')
    plt.yscale('log')

    #plt.ion()

    plt.scatter(GP, Eta, marker='*')
    for i in range(0, len(GP)):
        plt.annotate(str(i), (GP[i], Eta[i]))
    
    #Other: uncomment the following 2 lines.
    if not SPYDER:
        plt.draw()
        plt.pause(0.001)
    
    #Spyder: uncomment the following line.
    if SPYDER:
        plt.show()
    return None

def Average_mean(Eta):
    """Function that takes a pandas series and returns the mean and the error."""
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
    """Writes to a results.dat file the fitting results."""
    if not silent:
        print(name + ':', 'Intercept', aver, '+-', aver_err)

    with open('results.dat', 'a', encoding = 'utf-8') as fdest:
        fdest.write(name + ';' + str(aver) + ';' + str(aver_err) + ';' + extra +'\n')

        
def TreatFile(file, model='fitting'):
    """Treats an individual file using the model. At the moment, the model
    'fitting' is the best one."""
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

    if model == 'lin':
        print('Attempting to linearize the points')
        GP_arr = np.array(GP[initialp:finalp+1])  # +1 to include the last point the user wanted.
        Eta_arr = np.array(Eta[initialp:finalp+1])
        params = Parameters()
        # model: a + b * x
        params.add('a', value = 0, min = 0.0, max = 100)
        params.add('b', value = 0, min = 0.0, max = 10)
        
        from lmfit import minimize, Parameters, Parameter, report_fit
        
        result = minimize(fit_lin_min, params, args=(GP_arr, Eta_arr))
        final = Eta_arr + result.residual

        plt.plot(GP_arr, final)
        if not SPYDER:
            plt.draw()
            plt.pause(0.001)
        
        if Spyder:
            plt.show()
        

        report_fit(result)
        a = result.params['a'].value
        aerr = result.params['a'].stderr

        record(file, a, aerr)

    if model == 'mean':
        print('Averaging the points')
        aver, aver_err = Average_mean(Eta[initialp:finalp+1])
        record(file, aver, aver_err, extra=str(initialp)+str(finalp+1))
    
    if model == 'fitting':
        GP_arr = np.array(GP[initialp:finalp+1])
        Eta_arr = np.array(Eta[initialp:finalp+1])
        popt, pcov = curve_fit(fit_lin, GP_arr, Eta_arr, p0=(30, 0), 
                               bounds=(0, [1000., 0.0001])) #
        perr = np.sqrt(np.diag(pcov))

        a = popt[0]
        aerr = perr[0]
        record(file, a, aerr, extra=';first' + str(initialp) + 'final' + 
               str(finalp+1) + ';manual selection')
        plot_fitted_data(file, GP, Eta, popt[0], perr[0], initialp, finalp, method='manual')
    do_plot = 'Do you want to plot the data? y/[n]'
    if do_plot == 'y':
        plot_fitted_data(file, GP, Eta, a, aerr)
    return

    
# todo: Salvar nos plots os valores de a, aerr, e os pontos usados para fazer os fittings.
def plot_fitted_data(file, GP, Eta, a, aerr, p1, p2, method = '', do_save=True):
    """Plots the fitted data so that one can evaluate how good is the fit."""
    TEXT_X = 0.6
    TEXT_Y = 1.0
    if DEBUG:
        print('Debug: GP', GP, 'Eta', Eta)
    x = np.logspace(np.log10(GP[0]), np.log10(GP[-1]))
    if DEBUG:
        print('Debug: x', x)
    y = np.ones(len(x)) * a
    if DEBUG:
        print('Debug: y', y)
    yerr = np.ones(len(x)) * aerr
    
    if not SPYDER:
        plt.clf()
    
    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(GP, Eta)
    plt.errorbar(x, y, yerr=yerr)
    
    plt.annotate(str(p1), (GP[p1], Eta[p1]), color = 'red')
    plt.annotate(str(p2), (GP[p2], Eta[p2]), color = 'red')
    
    plt.figtext(TEXT_X - 0.3, TEXT_Y - 0.05, file)
    plt.figtext(TEXT_X - 0.3, TEXT_Y - 0.1, method)
    plt.figtext(TEXT_X, TEXT_Y - 0.05, 'a = ' + str(a), color = 'red')
    plt.figtext(TEXT_X, TEXT_Y - 0.10, 'aerr = ' + str(aerr), color = 'red')
    
    if do_save:
        plt.savefig(file[:-4] + '.png')
        print('Figure saved.')
    if not SPYDER:
        plt.draw()
        plt.pause(0.5)
        plt.clf()
    if SPYDER:
        plt.show()
    return


def automaticFitting(file, sorting = SORTING):
    """Goes through all the files, fits them and selects the best fit according to two algorithms.
    First, it selects two points, a beginning and an end point, the first starting at point 0
    and going to a third of the curve. The second, starting at points to the right,
    going until the middle of the curve.
    Then, it fits the data by fixing the slope at 0 and goes through every possible combination
    of the first and second points.
    It selects the data based on two criteria:
    1. sorting = 'by_error': finds the minimal error. Tends to select less points overall and 
        gives a fitting with a less than ideal representation overall.
    2. sorting = 'by_error_length': divides the error by how many points were used in the fit.
        May result in a higher overall error, but gives a better representation of the curve.
    """
    VISC_LIMIT = 100000. # Upper limit for the viscosity in the fitting.
    
    #segment = input('What is the segment that contains the flow curves for all the data?')
    GP, Eta = ExtractData(file)  #, FC_segment = segment)
    length = len(GP)
    
    fittings = []
    for first_point in range(0, length // 3, 1):
        for last_point in range(first_point + 3, length // 2, 1):
            GP_arr = np.array(GP[first_point:last_point + 1])
            Eta_arr = np.array(Eta[first_point:last_point + 1])
            popt, pcov = curve_fit(fit_lin, GP_arr, Eta_arr, p0=(30, 0), 
                                   bounds=(0, [VISC_LIMIT, 0.0001]))
            perr = np.sqrt(np.diag(pcov))
            fittings.append((first_point, last_point, popt, perr))
    #                           [0]           [1]      [2]   [3]
    #print('Debug: fittings: ', fittings)
    if sorting == 'by_error':
        fittings.sort(key = lambda x: x[3][0]) # selects the smallest error
    elif sorting == 'by_error_length':
        fittings.sort(key = lambda x: x[3][0] / (x[1]-x[0])) # selects the smallest error divided by
                                                             # the largest length
    else:
        print('Invalid sorting method. Please select change to a valid value.')
    a = fittings[0][2][0]  # popt
    aerr = fittings[0][3][0]  # perr
    if DEBUG:
        print('Debug: fittings_sorted: ', fittings)
        print('Debug: a: ', a)
        print('Debug: aerr: ', aerr)
    record(file, a, aerr, extra = ' first ' + str(fittings[0][0]) + ' last ' +
           str(fittings[0][1]) + ' sorting ' + sorting)
    #print('Debug: AutoFitting: GP', GP, 'Eta', Eta)
    plot_fitted_data(file, GP, Eta, a, aerr, p1=fittings[0][0], p2=fittings[0][1], method = sorting)
    return
            

#GP_pd = pd.Series(data = GP)
#Eta_pd = pd.Series(data = Eta)
# scipy.optimize.curve_fit(lambda x, m: m*x, xi, y)

#model = sm.OLS(Y, X_2)
#results = model.fit()
#par = results.params

if __name__ == '__main__':
    help = input('Do you want to read the help file? y/[n]')
    if help.lower() == 'y':
        help()
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