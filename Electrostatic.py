# Ur(r) = 4*pi*Ep*Ep0*a^2*Phi^2*exp(-k(r-a)/r)

def dbl_layer (Ep, Ep0, Phi, kappa, a):
	
	Value = 4*np.pi*Ep*Ep0*a**2*Phi**2*np.e**(-kappa*(r-a)/r)
	return Value

k = 1.38064852e-23 # m2 kg s-2 K-1
Ep = 1 		#medium permittivity
Ep0 = 8.85418e-2		# vacuum permittivity
Phi = 4		# surface potential
kappa = 5	#Debye screening length
a = 5e-9    #diameter
