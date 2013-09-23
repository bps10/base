from __future__ import division
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pylab as plt


from base import spectsens as s

class cone(object):
	''' cone_arma

		Program accompanying 'Simulating human cones from mid - mesopic
		up to high - photopic luminances' (J.H. van Hateren,  2006)

		Calculates normalized output of human cones (>response) to input time   
		series (with intensities in trolands at 1 ms resolution)

		Implemented according to a fast ARMA - algorithm
		Uses: steady.m

		Usage e.g.: load stimulus cone_arma plot(response)
		         see further README file and article

	    Author: J.H. van Hateren,  28 / 8 / 06
	    Subsequent development by B.P. Schmidt
'''

	def __init__(self):
		'''
		'''
		self._gen_data_container()
		self._gen_constants()
		self._gen_filter_constants()
		self.get_dark_values()
		
	def _gen_data_container(self):
		'''
		stim = stimulus (td,  normalized for bleaching)
		resp_b = fraction of pigment bleached
		resp_r = R*
		gain_r = fraction of ready pigment (R) 
		resp_ir = auxiliary variable (stim.R)
		resp_e = E*
		beta = free - space cGMP hydrolysis rate by PDE
		beta_e = effective cGMP hydrolysis rate by PDE
		resp_q = 1 / beta (main cone nonlinearity)
		tau_x = main adaptive cone time constant
		gain_x = gain by calcium feedback loop,  1 / alpha
		resp_x = cGMP
		resp_os = outer segment current
		resp_c = calcium concentration
		resp_vc= auxiliary variable (i_os / atten_i)
		resp_is = inner segment potential
		resp_im = auxiliary variable of membr. nonlinearity
		atten_i = attenuation by inner segment feedback loop
		'''

		self.data = {'stim': np.zeros(2), 'resp_b': np.zeros(2), 'resp_r': np.zeros(2),
					'gain_r': np.zeros(2), 'resp_ir': np.zeros(2), 'resp_e': np.zeros(2),     
					'beta': np.zeros(2), 'beta_e': np.zeros(2), 'resp_q': np.zeros(2),    
					'tau_x': np.zeros(2), 'gain_x': np.zeros(2), 'resp_x': np.zeros(2),    
					'resp_os': np.zeros(2), 'resp_c': np.zeros(2), 
					'resp_vc': np.zeros(2), 'resp_is': np.zeros(2),   
					'resp_im': np.zeros(2), 'atten_i': np.zeros(2), }

	def _gen_constants(self):
		'''
		nstep = time steps per ms default 10
		tau_r_ms = R* lifetime (np.expressed in ms)
		tau_r = R* lifetime np.expressed in time steps
		cn = normalization constant for cone bleaching (4.1e - 9 if tau_r = 3.4 ms) 
		tau_b0 = time constant of bleaching recovery
		rk_b = parameter of bleaching recovery
		tau_e = E*  lifetime
		c_beta = dark PDE activity ( = 1 / tau_D)
		rk_beta = E*  dependence of PDE activity
		beta_e_max = parameter diffusion - limited cGMP hydrolysis
		rnx = apparent Hill coefficient CNG channels
		rnc = Hill coefficient GC activation
		tau_c = time constant of Ca2 +  extrusion
		a_c = scaling constant of GC activation
		tau_vc = membrane time constant
		gamma_is = parameter of membrane nonlinearity
		tau_is = time constant of membrane nonlinearity
		a_is = parameter of membrane nonlinearity
		'''
		self.const = {}
		self.const['nstep'] = 10               
		self.const['tau_r_ms'] = 3.4          
		self.const['tau_r'] = 3.4 * self.const['nstep'] 
		self.const['cn'] = 4.1e-9 * self.const['tau_r_ms'] / 3.4
		self.const['tau_b0'] = 25000 * self.const['nstep']  
		self.const['rk_b'] = 0.2               
		self.const['tau_e'] = 8.7 * self.const['nstep']     
		self.const['c_beta'] = 2.8e-3          
		self.const['rk_beta'] = 1.4e-4         
		self.const['beta_e_max'] = 4           
		self.const['rnx'] = 1                  
		self.const['rnc'] = 4                 
		self.const['tau_c'] = 3 * self.const['nstep']    
		self.const['a_c'] = 0.23               
		self.const['tau_vc'] = 4 * self.const['nstep']  
		self.const['gamma_is'] = 0.7           
		self.const['tau_is'] = 90 * self.const['nstep']
		self.const['a_is'] = 2.9e-2            

	def _gen_filter_constants(self):
		'''
		'''
		self.k = {}
		self.k['f1_tau_r'] = np.exp( - 1 / self.const['tau_r'])
		self.k['f2_tau_r'] = (self.const['tau_r'] - (1 + self.const['tau_r']) *
									self.k['f1_tau_r'])
		self.k['f3_tau_r'] = (1 - self.const['tau_r'] + self.const['tau_r'] *
									self.k['f1_tau_r'])
		      
		self.k['f1_tau_e'] = np.exp( - 1 / self.const['tau_e'])
		self.k['f2_tau_e'] = (self.const['tau_e'] - (1 + self.const['tau_e']) * 
									self.k['f1_tau_e'])
		self.k['f3_tau_e'] = (1 - self.const['tau_e'] + self.const['tau_e'] * 
									self.k['f1_tau_e'])
		      
		self.k['f1_tau_c'] = np.exp( - 1 / self.const['tau_c'])
		self.k['f2_tau_c'] = (self.const['tau_c'] - (1 + self.const['tau_c']) * 
									self.k['f1_tau_c'])
		self.k['f3_tau_c'] = (1 - self.const['tau_c'] + self.const['tau_c'] * 
									self.k['f1_tau_c'])
		      
		self.k['f1_tau_vc'] = np.exp( - 1 / self.const['tau_vc'])
		self.k['f2_tau_vc'] = (self.const['tau_vc'] - (1 + self.const['tau_vc']) * 
									self.k['f1_tau_vc'])
		self.k['f3_tau_vc'] = (1 - self.const['tau_vc'] + self.const['tau_vc'] * 
									self.k['f1_tau_vc'])
		      
		self.k['f1_tau_is'] = np.exp( - 1 / self.const['tau_is'])
		self.k['f2_tau_is'] = self.const['a_is'] * (self.const['tau_is'] - 
								(1 + self.const['tau_is']) * self.k['f1_tau_is'])
		self.k['f3_tau_is'] = self.const['a_is'] * (1 - self.const['tau_is'] + 
								self.const['tau_is'] * self.k['f1_tau_is'])            
	
	def set_stimulus(self, stimulus):
		'''
		'''
		self.stimulus = stimulus
		# use first value of stimulus as adaption state
		self.get_adaptive_states()

	def get_dark_values(self):
		'''
		get dark value of resp_is
		'''
		self.dark = {}
		self.td = 0
		self.dark['resp_c'] = fsolve(self.steady, 1)[0]
		self.dark['beta'] = self.const['c_beta']
		self.dark['beta_e'] = self.dark['beta'] / (1 + self.dark['beta'] /
								self.const['beta_e_max'])
		self.dark['resp_q'] = 1 / self.dark['beta_e']
		self.dark['gain_x'] = 1 / (1 + (self.const['a_c'] * 
								self.dark['resp_c']) ** self.const['rnc'])
		self.dark['resp_x'] = self.dark['gain_x'] * self.dark['resp_q']
		self.dark['resp_os'] = self.dark['resp_x'] ** self.const['rnx']
		self.dark['resp_is'] = ((self.dark['resp_os'] / self.const['a_is']) ** 
								(1 / (1 + self.const['gamma_is'])))

	def get_adaptive_states(self):
		'''determine adaptive state
		'''

		nprev = 1
		# assumed steady - state intensity,  used by 'steady'
		self.td = self.stimulus[0]  
		self.data['resp_c'][nprev] = fsolve(self.steady, 1)[0]
		self.data['stim'][nprev] = self.td

		if self.data['stim'][nprev] == 0:
			self.data['resp_b'][nprev] = 0
			self.data['resp_r'][nprev] = 0
			self.data['gain_r'][nprev] = 1
			self.data['resp_ir'][nprev] = 0
			self.data['resp_e'][nprev] = 0

		else:
			stimn = self.data['stim'][nprev] * self.const['cn']
			self.data['resp_b'][nprev] = (0.5 * (1 - self.const['rk_b'] - 
								self.const['tau_r'] / self.const['tau_b0'] * 
								self.const['rk_b'] * (stimn + 1) / 
								stimn + np.sqrt((1 - self.const['rk_b'] - 
									self.const['tau_r'] / self.const['tau_b0'] * 
									self.const['rk_b'] * 
								(stimn + 1) / stimn) ** 2 + 4 * self.const['rk_b'])))

			self.data['resp_r'][nprev] = ((1 - self.data['resp_b'][nprev]) * 
								self.data['stim'][nprev] / (1 + stimn))
			self.data['gain_r'][nprev] = (1 - self.data['resp_b'][nprev] - 
								self.const['cn'] * self.data['resp_r'][nprev])
			self.data['resp_ir'][nprev] = (self.data['gain_r'][nprev] * 
							self.data['stim'][nprev])
			self.data['resp_e'][nprev] = self.data['resp_r'][nprev]

		self.data['beta'][nprev] = (self.const['c_beta'] + self.const['rk_beta'] * 
									self.data['resp_e'][nprev])
		self.data['beta_e'][nprev] = (self.data['beta'][nprev] / (1 + 
							self.data['beta'][nprev] / self.const['beta_e_max']))
		self.data['resp_q'][nprev] = 1 / self.data['beta_e'][nprev]
		# express tau_x in time steps
		self.data['tau_x'][nprev] = self.data['resp_q'][nprev] * self.const['nstep'] 
		self.data['gain_x'][nprev] = (1 / (1 + (self.const['a_c'] * 
							self.data['resp_c'][nprev]) ** self.const['rnc']))

		self.data['resp_x'][nprev] = (self.data['gain_x'][nprev] * 
							self.data['resp_q'][nprev])
		self.data['resp_os'][nprev] = self.data['resp_x'][nprev] ** self.const['rnx']
		self.data['resp_vc'][nprev] = ((self.data['resp_os'][nprev] / 
							self.const['a_is']) ** (1 / (1 + self.const['gamma_is'])))
		self.data['resp_is'][nprev] = self.data['resp_vc'][nprev]
		self.data['resp_im'][nprev] = (self.data['resp_is'][nprev] ** 
							self.const['gamma_is'])
		self.data['atten_i'][nprev] = self.const['a_is'] * self.data['resp_im'][nprev]
		

	def simulate(self):
		'''
		'''
		nstim = len(self.stimulus)
		response = np.zeros(nstim)

		for istim in range(0, nstim):
			for istep in range(0, self.const['nstep']):
			   
				if istim == 0 and istep == 0:
					ncurr = 0             # current element
					nprev = 1             # previous element
				else:
					nkl = ncurr           # swap ncurr / nprev
					ncurr = nprev
					nprev = nkl

				self.data['stim'][ncurr] = self.stimulus[istim] 

				self.data['resp_ir'][ncurr] = (self.data['gain_r'][nprev] * 
							self.data['stim'][ncurr])
				
				self.data['resp_r'][ncurr] = (self.k['f1_tau_r'] * 
							self.data['resp_r'][nprev] + self.k['f2_tau_r'] * 
							self.data['resp_ir'][nprev] + self.k['f3_tau_r'] * 
							self.data['resp_ir'][ncurr])
				
				tau_b = self.const['tau_b0'] * (self.data['resp_b'][nprev] + 
							self.const['rk_b']) / self.const['rk_b']
				
				gain_b = self.const['cn'] * tau_b / self.const['tau_r']
				
				f1_tau_b = np.exp( - 1 / tau_b)
				f2_tau_b = gain_b * (tau_b - (1 + tau_b) * f1_tau_b)
				f3_tau_b = gain_b * (1 - tau_b + tau_b * f1_tau_b)
				
				self.data['resp_b'][ncurr] = (f1_tau_b * self.data['resp_b'][nprev] + 
							f2_tau_b * self.data['resp_r'][nprev] + 
							f3_tau_b * self.data['resp_r'][ncurr])
				
				self.data['gain_r'][ncurr] = (1 - self.data['resp_b'][ncurr] -
							self.const['cn'] * self.data['resp_r'][ncurr])
				 
				self.data['resp_e'][ncurr] = (self.k['f1_tau_e'] * 
							self.data['resp_e'][nprev] + self.k['f2_tau_e'] * 
							self.data['resp_r'][nprev] + self.k['f3_tau_e'] * 
							self.data['resp_r'][ncurr])

				self.data['beta'][ncurr] = (self.const['c_beta'] + 
							self.const['rk_beta'] * self.data['resp_e'][ncurr])
				self.data['beta_e'][ncurr] = (self.data['beta'][ncurr] / 
							(1 + self.data['beta'][ncurr] / self.const['beta_e_max']))
				self.data['resp_q'][ncurr] = 1 / self.data['beta_e'][ncurr]

				# express tau_x in time steps
				self.data['tau_x'][ncurr] = (self.data['resp_q'][ncurr] * 
							self.const['nstep'])
				f1_tau_x = np.exp( - 1 / self.data['tau_x'][ncurr])
				f2_tau_x = (self.data['tau_x'][ncurr] - 
							(1 + self.data['tau_x'][ncurr]) * f1_tau_x)
				f3_tau_x = (1 - self.data['tau_x'][ncurr] + 
							self.data['tau_x'][ncurr] * f1_tau_x)

				self.data['resp_x'][ncurr] = (f1_tau_x * self.data['resp_x'][nprev] + 
							self.data['gain_x'][nprev] * f2_tau_x * 
							self.data['resp_q'][nprev] + self.data['gain_x'][nprev] * 
							f3_tau_x * self.data['resp_q'][ncurr])

				
				self.data['resp_os'][ncurr] = (self.data['resp_x'][ncurr] ** 
							self.const['rnx'])
				          
				self.data['resp_c'][ncurr] = (self.k['f1_tau_c'] * 
							self.data['resp_c'][nprev] + self.k['f2_tau_c'] * 
							self.data['resp_os'][nprev] + self.k['f3_tau_c'] * 
							self.data['resp_os'][ncurr])
				 
				self.data['gain_x'][ncurr] = (1 / (1 + (self.const['a_c'] * 
							self.data['resp_c'][ncurr]) ** self.const['rnc']))
				   
				self.data['resp_vc'][ncurr] = (self.data['resp_os'][ncurr] / 
							self.data['atten_i'][nprev] )
				
				self.data['resp_is'][ncurr] = (self.k['f1_tau_vc'] * 
							self.data['resp_is'][nprev] + self.k['f2_tau_vc'] * 
							self.data['resp_vc'][nprev] + self.k['f3_tau_vc'] * 
							self.data['resp_vc'][ncurr])
				 
				self.data['resp_im'][ncurr] = (self.data['resp_is'][ncurr] ** 
							self.const['gamma_is'])
				   
				self.data['atten_i'][ncurr] = (self.k['f1_tau_is'] * 
							self.data['atten_i'][nprev] + self.k['f2_tau_is'] * 
							self.data['resp_im'][nprev] + self.k['f3_tau_is'] * 
							self.data['resp_im'][ncurr])      
				
			resp_norm = ((self.data['resp_is'][ncurr] - self.dark['resp_is']) /
								self.dark['resp_is'] )
			response[istim] = resp_norm

		return response

	def steady(self, x):
		''' 
			finds steady - state: steady = 0 if x = steady - state of resp_c

			Function accompanying 'Simulating human cones from mid - mesopic
			up to high - photopic luminances' (J.H. van Hateren, 2006)

			Used by: cone_arma.m

			Author: J.H. van Hateren, 28 / 8 / 06
			Adapted by B.P. Schmidt
		'''
		stim = self.td
		if stim == 0:
			resp_e = 0
		else:
		 	stimn = self.td * self.const['cn']
		 	resp_b = (0.5 * (1 - self.const['rk_b'] - self.const['tau_r'] / 
		 				self.const['tau_b0'] * self.const['rk_b'] * (stimn + 1) / 
		 				stimn + np.sqrt((1 - self.const['rk_b'] - 
		 				self.const['tau_r'] / self.const['tau_b0'] * 
		 				self.const['rk_b'] * (stimn + 1) / stimn) ** 2 + 4 * 
		 				self.const['rk_b'])))
		 	resp_r = (1 - resp_b) * stim / (1 + stimn)
		 	resp_e = resp_r

		beta = self.const['c_beta'] + self.const['rk_beta'] * resp_e
		beta_e = beta / (1 + beta / self.const['beta_e_max'])

		return (x  -  ((1  /  (1  +  (self.const['a_c']  *  x)  **  
				self.const['rnc'])) / beta_e)  **  self.const['rnx'])


def plot_cone():
	'''
	'''
	
	'''sensitivity = s.neitz(LambdaMax=559, OpticalDensity=0.4, LOG=False,
            	StartWavelength=wvlen, EndWavelength=wvlen, 
            	resolution=1, EXTINCTION=False)'''
	stimulus = np.genfromtxt('base/cones/stimulus.txt')
	c = cone()
	c.set_stimulus(stimulus)
	response = c.simulate()

	fig  = plt.figure(figsize=(10, 8))
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	pf.AxisFormat()
	pf.TufteAxis(ax1, ['left'], [5, 5])
	pf.TufteAxis(ax2, ['left', 'bottom'], [5, 5])

	ax1.semilogy(stimulus, 'k')
	ax2.plot(response, 'k')

	ax1.set_ylabel('luminance (td)')
	ax2.set_ylabel('normalized response')
	ax2.set_xlabel('time')

	plt.tight_layout()
	plt.show()

def plot_frequencies():
	'''
	'''
	x = np.arange(0, np.pi, 0.001)
	sinewave = (1000 * np.sin(x * 1)) + 5000
	c = cone()
	c.set_stimulus(sinewave)
	response1 = c.simulate()

	sinewave = (1000 * np.sin(x * 25)) + 5000
	c = cone()
	c.set_stimulus(sinewave)
	response2 = c.simulate()

	sinewave = (1000 * np.sin(x * 50)) + 5000
	c = cone()
	c.set_stimulus(sinewave)
	response3 = c.simulate()	

	sinewave = (1000 * np.sin(x * 150)) + 5000
	c = cone()
	c.set_stimulus(sinewave)
	response4 = c.simulate()


	fig  = plt.figure(figsize=(10, 8))
	ax1 = fig.add_subplot(411)
	ax2 = fig.add_subplot(412)
	ax3 = fig.add_subplot(413)
	ax4 = fig.add_subplot(414)

	pf.AxisFormat()
	pf.TufteAxis(ax1, ['left'], [5, 5])
	pf.TufteAxis(ax2, ['left'], [5, 5])
	pf.TufteAxis(ax3, ['left'], [5, 5])
	pf.TufteAxis(ax4, ['left'], [5, 5])

	ax1.plot(x, response1, 'k')
	ax2.plot(x, response2, 'k')
	ax3.plot(x, response3, 'k')
	ax4.plot(x, response4, 'k')



	ax1.set_ylabel('luminance (td)')
	#ax2.set_ylabel('normalized response')
	#ax3.set_ylabel('normalized response')
	#ax4.set_ylabel('normalized response')
	ax4.set_xlabel('time')

	#ax4.set_xlim([min(x), max(x)])

	plt.tight_layout()
	plt.show()


def temporal_sensitivity(background_illum=1000, contrast=100):
	'''
	'''
	
	x = np.arange(0, np.pi * 2, 0.001)
	amp = []
	freqs = 10 ** np.arange(0, 3, 0.1)

	k = background_illum * (contrast / 100.)
	for freq in freqs:
		sinewave = (k * np.sin(x * freq)) + background_illum
		c = cone()
		c.set_stimulus(sinewave)
		response = c.simulate()
		amp.append(abs(np.max(response) - np.min(response)))
	amp = np.asarray(amp)

	fig  = plt.figure()
	ax1 = fig.add_subplot(111)

	pf.AxisFormat()
	pf.TufteAxis(ax1, ['left', 'bottom'], [5, 5])

	ax1.semilogx(freqs, amp / np.max(amp), 'ko-')

	ax1.set_xlabel('frequency (Hz)')
	ax1.set_ylabel('sensitivity')

	plt.tight_layout()
	plt.show()



if __name__ == '__main__':

	from base import plot as pf
	temporal_sensitivity()
	#plot_frequencies()
	#plot_cone()