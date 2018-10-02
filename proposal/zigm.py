def zigm(lam_obs, redshift, method="Meiksin", sep_lines_cont=False):
	"""
	Compute the Hydrogen attenuation due to intervening clouds

	Computes the stochastic Hydrogen attenuation due to intervening clouds

	By default the prescription of Meiksin, 2006, MNRAS 365, 807, is used.

	User can also select prescription of Madau 1995, ApJ, 441, 18.

	Returns the spectrum of tau values
	Transmission can be retrieved as T = exp(-tau)
	
	by D. Watson 2014-04-23 (DARK/U. Copenhagen)

	Args:
		lam_obs = input observed wavelength array (angstroms)
		redshift = source redshift
		method = "Madau" or "Meiksin" prescriptions
		sep_lines_cont = True/False to return tau components separately
	Returns:
		tau_sum = spectrum of opacities by observed wavelength
		(tau_continuum = opacities due to Lyman continuum)
		(tau_lines = opacities due to line absorption)
	Raises:
		MethodException: if method name is not "Meiksin" or "Madau"
	"""  

	if method == "Meiksin":
		tau_spec, tau_c, tau_l = fmmeiksin(lam_obs,redshift)
	elif method == "Madau":
		tau_spec, tau_c, tau_l = fmmadau(lam_obs,redshift)
	else:
		raise MethodException("Choose a method, either \"Meiksin\" or \"Madau\", doofus!")
	
	if sep_lines_cont!=False:
		return tau_spec, tau_c, tau_l
	else:
		return tau_spec


class MethodException(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg


def fmmeiksin(lam_obs,z):
	"""
	Compute the hydrogen attenuation due to intervening clouds
	using the prescription of Meiksin, 2006, MNRAS 365, 807

	See wrapper function zigm()

	Returns the spectrum of tau values
	Transmission can be retrieved as T = exp(-tau)
	
	by D. Watson 2014-04-23 (DARK/U. Copenhagen)

	Args:
		lam_obs = input observed wavelength array (angstroms)
		redshift = source redshift
	Returns:
		tau_sum = spectrum of opacities by observed wavelength
		tau_continuum = opacities due to Lyman continuum
		tau_lines = opacities due to line absorption
	"""  

	import numpy as np
	from scipy.special import gammaincc
	from scipy.misc import factorial
	from scipy.constants import Rydberg


	# Ly_series_coeffs contains the ratio tau_n to tau_alpha (alpha is n=2) for n=3 to 9
	# Little kludge on the alpha coeff (0.5) to make the matrix multiplication below work right
	

	Ly_series_coeffs = np.lib.pad(np.array([0.5, 0.348, 0.179, 0.109, 0.0722, 0.0508, 0.0373, 0.0283]), (0,21), mode='constant')
	
	lam_Ly_lim = 1.0e10/Rydberg # Ly limit wavelength in angstroms
	n = np.arange(2,31,1)
	lam_n = lam_Ly_lim/(1.0-(n**-2.0)) # Wavelength of line n
	lam_rat = np.divide.outer(lam_obs,lam_n)

	zone = z + 1.0
	zn = lam_rat - 1.0
	# Lyman series attenuation
	if z < 4.0:
		coeff = np.array([0.00211, 3.7]) # Meiksin Eq. 2
	else:
		coeff = np.array([0.00058, 4.5]) # Meiksin Eq. 3
	tau_alpha = coeff[0]*(zn**coeff[1]) # Meiksin Eqs. 2 & 3

	third = 1.0/3.0
	sixth = 1.0/6.0
	powers = np.zeros((len(n),2))
	powers[0,:] = 0.0
	powers[1:8,:] = np.array([[third,sixth], \
						[third, sixth], \
						[third, sixth], \
						[third, third], \
						[third, third], \
						[third, third], \
						[third, third]])

	# tau_lines should be a 8*len(lam_rat) array
	tau_lowlines = np.zeros((len(lam_obs),8))
	tau_higherlines = np.zeros((len(lam_obs),21))
#	tau_lowlines[:,0:2] = np.zeros(tau_lowlines[:,0:2].shape)
	tau_lowlines[:,:8] = tau_alpha[:,:8]*Ly_series_coeffs[:8]*( \
		np.power(0.25*(lam_rat[:,:8]*(lam_rat[:,:8]<4)),powers[:8,0]) + \
		np.power(0.25*(lam_rat[:,:8]*(lam_rat[:,:8]>=4)),powers[:8,1]) \
		)

    # Create the tau for the remaining 20 lines
	tau_higherlines = np.multiply.outer( tau_lowlines[:,7], 720/(n[8:]*((n[8:]**2.0) - 1.0)) )
    # Add the remaining 20 lines
	tau_lines_nowav = np.hstack((tau_lowlines, tau_higherlines))

	tau_lines = (np.less_equal.outer(lam_obs, lam_n*zone)*tau_lines_nowav).sum(axis=1) # Create a binary array mask where we check whether the lines have any effect on those wavelengths and then sum the tau's
	tau_lines[tau_lines<0] = 0.0

	zLone = 1.0 + (lam_obs/lam_Ly_lim - 1.0)

	tau_igm_nowav = 0.805*(zLone**3.0)*((1.0/zLone) - (1.0/zone)) # IGM Meiksin Eq. 5

	tau_igm = tau_igm_nowav*(lam_obs<(zone*lam_Ly_lim)) # Opacity only applies below the Lyman limit
	tau_igm[tau_igm<0] = 0.0


	# Meiksin Eq. 7
	bet = 1.5
	betmone = bet - 1.0
	gam = 1.5
	N0 = 0.25
	m = np.arange(10) # LLS summation index (10 is enough says Meiksin)

	term1 = gammaincc(2.0-bet,1.0) - 1.0/np.e - ((betmone/(m-betmone))*(((-1)**m)/factorial(m))).sum()
	term2 = (zone**((-3.0*betmone)+gam+1.0)) * np.power(zLone,3.0*betmone) - np.power(zLone,gam+1.0)

	term3 = ((betmone/((3.0*m - gam - 1.0)*(m-betmone)))*((-1)**m)/factorial(m))
	term4 = ((zone**(gam + 1.0 - 3.0*m))*np.power.outer(zLone,3*m)).T - zLone**(gam+1.0)

	tau_LLS_nowav = (N0/(4 + gam - (3.0*bet)))*term1*term2 - (N0*term3*term4.T).sum(axis=1)

	tau_LLS = tau_LLS_nowav*(lam_obs<(zone*lam_Ly_lim)) # Opacity only applies below the Lyman limit
	tau_LLS[tau_LLS<0] = 0.0

	tau_cont = tau_igm + tau_LLS

	tau_total = (tau_lines + tau_cont)

	return tau_total, tau_cont, tau_lines






def fmmadau(lam_obs,z):
	"""
	Compute the hydrogen attenuation due to intervening clouds
	using the prescription of Madau 1995, ApJ, 441, 18.

	See wrapper function zigm()

	Returns the spectrum of tau values
	Transmission can be retrieved as T = exp(-tau)
	
	by D. Watson 2014-04-23 (DARK/U. Copenhagen)

	Args:
		lam_obs = input observed wavelength array (angstroms)
		redshift = source redshift
	Returns:
		tau_sum = spectrum of opacities by observed wavelength
		tau_continuum = opacities due to Lyman continuum
		tau_lines = opacities due to line absorption
	"""  

	import numpy as np
	from scipy.special import gammaincc
	from scipy.misc import factorial
	from scipy.constants import Rydberg

	a_metal=0.0017
	zone = z + 1.0
	Ly_series_coeffs = np.array([0.0036, 0.0017, 0.0011846, \
						0.0009410, 0.0007960, 0.0006967, 0.0006236, \
						0.0005665, 0.0005200, 0.0004817, 0.0004487, \
						0.0004200, 0.0003947, 0.0003720, 0.0003520, \
						0.0003334, 0.00031644])
	lam_Ly_lim = 1e10/Rydberg # Ly limit wavelength in angstroms
	n = np.arange(2,19,1)
	lam_n = lam_Ly_lim/(1.0-(n**-2.0)) # Wavelength of line n
	lam_rat = np.divide.outer(lam_obs,lam_n)
	xc = lam_obs/lam_Ly_lim
	tau_lines_nowav = Ly_series_coeffs*(lam_rat**3.46) + \
					np.vstack( (a_metal*((lam_obs/lam_n[2])**1.68), np.zeros((len(n)-1,len(lam_obs)))) ).T

	tau_lines = (np.less_equal.outer(lam_obs, lam_n*zone)*tau_lines_nowav).sum(axis=1) # Create a binary array mask where we check whether the lines have any effect on those wavelengths and then sum the tau's
	tau_lines[tau_lines<0] = 0.0


	tau_cont_nowav =  (0.25  * (xc**3.0) * ((zone**( 0.46)) - (xc**( 0.46)))) \
					+ (9.4   * (xc**1.5) * ((zone**( 0.18)) - (xc**( 0.18)))) \
					+ (0.7   * (xc**3.0) * ((zone**(-1.32)) - (xc**(-1.32)))) \
					- (0.023 *             ((zone**( 1.68)) - (xc**( 1.68))))

	tau_cont = tau_cont_nowav*(lam_obs<(zone*lam_Ly_lim)) # Opacity only applies below the Lyman limit
	tau_cont[tau_cont<0] = 0.0

	tau_total = tau_lines + tau_cont

	return tau_total, tau_cont, tau_lines




