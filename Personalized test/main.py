import glob
import random
import numpy as np

names = [name.rstrip() for name in open('Random names.txt')]

fhand_q1 = open('Q1.txt')

mode = 'transcribing'
q_body = ''
params = []

for line in fhand_q1:
    #print('Debug:', mode)
    #print(line.startswith('##'))
    if line.startswith('COM'):
        continue
    if line.startswith('##'):
        mode = 'options'
        continue
    if mode == 'transcribing':
        q_body = q_body + line
    
    if mode == 'options':
        if line.startswith('%n '):
            range_par = tuple(line.strip('%n').rstrip().split(','))
            start = float(range_par[0])
            end = float(range_par[1])
            
            try:
                step = float(range_par[2])
            except IndexError:
                step = 1
            range = np.arange(start, end, step)
            params.append(range)
        if line.startswith('%s'):
            params.append(line.strip('%s').rstrip().split(','))

choices = [item[3] for item in params]
#print('Debug: choices', choices)
#print('Debug: choices tuple', tuple(choices))
#print('Body:', q_body.format(choices[0], choices[1], choices[2], choices[3]))
print('Body:', q_body.format(*tuple(choices)))
#print(params)