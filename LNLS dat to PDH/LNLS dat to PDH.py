import glob
preamble1 = """
SAXS BOX                                                                        
     """
preamble2 = """         0         0         0         0         0         0         0 
  0.000000E+00   3.068000E+02   0.000000E+00   1.000000E+00   1.542000E-01
  2.500000E+01   1.062232E+00   0.000000E+00   0.000000E+00   0.000000E+00 \n"""

arquivos = glob.glob('*.dat')
for arquivo in arquivos:
    dados = open(arquivo,'r').read().split('\n')
    d_com_erro = [dado + ' 0 \n' for dado in dados]
    d_com_erro = d_com_erro[:-1]
    with open(arquivo[:-4] + "_conv.dat",'w') as fhand:
        fhand.write(preamble1 + str(len(d_com_erro)) + preamble2)
        for line in d_com_erro:
            fhand.write(line)