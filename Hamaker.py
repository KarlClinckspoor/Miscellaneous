#constants

h_bar = 1.0545718e-34 # m2 kg / s Reduced Planck constant
h = 6.62607004e-34 # m2 kg / s Planck constant
k = 1.38064852e-23 # m2 kg s-2 K-1 Boltzmann constant

#H = 3/4 kT (Ea-Es/(Ea+Es))^2+3/(16*2^(1/2))h_bar*w*(na2-ns2)/(na2-ns2)/3/2
def Hamaker (Eagg, Esolv, nagg, nsolv, T):
	omega = 2.4e16 #rad/s
	H = 3/4*k*T*((Eagg-Esolv)/(Eagg+Esolv))**2+3/(16*2**(1/2))*h_bar*omega*((nagg**2-nsolv**2)/(nagg**2+nsolv**2)**(3/2))
	return H
	
def Hamaker_sep(Eagg, Esolv, nagg, nsolv, T):
	nu = 3e15 #s-1
	H_epsilon = 3/4*k*T*((Eagg-Esolv)/(Eagg+Esolv))**2
	H_refractive = 3/(16*2**(1/2))*h*nu*((nagg**2-nsolv**2)**2/(nagg**2+nsolv**2)**(3/2))
	H_total = H_epsilon + H_refractive
	return H_total, H_epsilon, H_refractive
'''
Eagg = float(input('Enter Diel Const of Agg: '))
Esolv = float(input('Enter Diel Const of Solv: '))
nagg = float(input('Enter refractive index agg: '))
nsolv = float(input('Enter refractive index solv: '))
T = float(input('Enter temperature:'))
'''
Eagg = 26.18
Esolv = 80.37
nagg = 1.366
nsolv = 1
T = 298
properties = list()
concentrations = list()


fhand = open('Glicerina')
for line in fhand:
	(comp_glic, n_glic, ep_glic) = line.split()
	properties.append((Eagg, float(ep_glic), nagg, float(n_glic), T))
	concentrations.append(float(comp_glic))

fhand.close()

count = 0

for item in properties:
	total, epsilon, refractive = Hamaker_sep(item[0], item[1], item[2], item[3], item[4])
	kT = k*T
	print (concentrations[count], '% of glycerol:', total,'J:', epsilon, 'J+', refractive, 'J')
	count += 1