from __future__ import division
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pylab as plt

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
'''

	def __init__(self):
		'''
		'''
		self._gen_data_container()
		self._gen_constants()
		self._gen_time_constants()
		self.get_dark_values()
		self.stimulus = self.load_stimulus()
		self.get_adaptive_states()
		
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

		self.data = {}
		self.data['stim'] = np.zeros(2)      
		self.data['resp_b'] = np.zeros(2)    
		self.data['resp_r'] = np.zeros(2)     
		self.data['gain_r'] = np.zeros(2)    
		self.data['resp_ir'] = np.zeros(2)   
		self.data['resp_e'] = np.zeros(2)     
		self.data['beta'] = np.zeros(2)      
		self.data['beta_e'] = np.zeros(2)    
		self.data['resp_q'] = np.zeros(2)    
		self.data['tau_x'] = np.zeros(2)     
		self.data['gain_x'] = np.zeros(2)    
		self.data['resp_x'] = np.zeros(2)    
		self.data['resp_os'] = np.zeros(2)   
		self.data['resp_c'] = np.zeros(2)    
		self.data['resp_vc'] = np.zeros(2)   
		self.data['resp_is'] = np.zeros(2)   
		self.data['resp_im'] = np.zeros(2)   
		self.data['atten_i'] = np.zeros(2)   

	def _gen_constants(self):
		'''
		'''
		self.const = {}
		self.const['nstep'] = 10               # time steps per ms default 10
		self.const['tau_r_ms'] = 3.4           # R* lifetime (np.expressed in ms)
		self.const['tau_r'] = 3.4 * self.const['nstep']   # R* lifetime np.expressed in time steps
		self.const['cn'] = 4.1e-9 * self.const['tau_r_ms'] / 3.4 # normalization constant for cone - 
		                         			   # bleaching (4.1e - 9 if tau_r = 3.4 ms)
		self.const['tau_b0'] = 25000 * self.const['nstep']   # time constant of bleaching recovery
		self.const['rk_b'] = 0.2               # parameter of bleaching recovery
		self.const['tau_e'] = 8.7 * self.const['nstep']      # E*  lifetime
		self.const['c_beta'] = 2.8e-3          # dark PDE activity ( = 1 / tau_D)
		self.const['rk_beta'] = 1.4e-4         # E*  dependence of PDE activity
		self.const['beta_e_max'] = 4           # parameter diffusion - limited cGMP hydrolysis
		self.const['rnx'] = 1                  # apparent Hill coefficient CNG channels
		self.const['rnc'] = 4                  # Hill coefficient GC activation
		self.const['tau_c'] = 3 * self.const['nstep']    # time constant of Ca2 +  extrusion
		self.const['a_c'] = 0.23               # scaling constant of GC activation
		self.const['tau_vc'] = 4 * self.const['nstep']  # membrane time constant
		self.const['gamma_is'] = 0.7           # parameter of membrane nonlinearity
		self.const['tau_is'] = 90 * self.const['nstep']  # time constant of membrane nonlinearity
		self.const['a_is'] = 2.9e-2            # parameter of membrane nonlinearity

	def _gen_time_constants(self):
		'''
		'''
		self.time_k = {}
		self.time_k['f1_tau_r'] = np.exp( - 1 / self.const['tau_r'])
		self.time_k['f2_tau_r'] = (self.const['tau_r'] - (1 + self.const['tau_r']) *
									self.time_k['f1_tau_r'])
		self.time_k['f3_tau_r'] = (1 - self.const['tau_r'] + self.const['tau_r'] *
									self.time_k['f1_tau_r'])
		      
		self.time_k['f1_tau_e'] = np.exp( - 1 / self.const['tau_e'])
		self.time_k['f2_tau_e'] = (self.const['tau_e'] - (1 + self.const['tau_e']) * 
									self.time_k['f1_tau_e'])
		self.time_k['f3_tau_e'] = (1 - self.const['tau_e'] + self.const['tau_e'] * 
									self.time_k['f1_tau_e'])
		      
		self.time_k['f1_tau_c'] = np.exp( - 1 / self.const['tau_c'])
		self.time_k['f2_tau_c'] = (self.const['tau_c'] - (1 + self.const['tau_c']) * 
									self.time_k['f1_tau_c'])
		self.time_k['f3_tau_c'] = (1 - self.const['tau_c'] + self.const['tau_c'] * 
									self.time_k['f1_tau_c'])
		      
		self.time_k['f1_tau_vc'] = np.exp( - 1 / self.const['tau_vc'])
		self.time_k['f2_tau_vc'] = (self.const['tau_vc'] - (1 + self.const['tau_vc']) * 
									self.time_k['f1_tau_vc'])
		self.time_k['f3_tau_vc'] = (1 - self.const['tau_vc'] + self.const['tau_vc'] * 
									self.time_k['f1_tau_vc'])
		      
		self.time_k['f1_tau_is'] = np.exp( - 1 / self.const['tau_is'])
		self.time_k['f2_tau_is'] = self.const['a_is'] * (self.const['tau_is'] - 
								(1 + self.const['tau_is']) * self.time_k['f1_tau_is'])
		self.time_k['f3_tau_is'] = self.const['a_is'] * (1 - self.const['tau_is'] + 
								self.const['tau_is'] * self.time_k['f1_tau_is'])            
	
	def load_stimulus(self):
		'''
		'''
		return np.genfromtxt('stimulus.txt')

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
		self.td = self.stimulus[0]  # assumed steady - state intensity,  used by 'steady'
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
			self.data['resp_ir'][nprev] = self.data['gain_r'][nprev] * self.data['stim'][nprev]
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
		#print self.data['gain_x'], self.const['a_c'], self.data['resp_c']
		self.data['resp_x'][nprev] = self.data['gain_x'][nprev] * self.data['resp_q'][nprev]
		self.data['resp_os'][nprev] = self.data['resp_x'][nprev] ** self.const['rnx']
		self.data['resp_vc'][nprev] = ((self.data['resp_os'][nprev] / self.const['a_is']) ** 
											(1 / (1 + self.const['gamma_is'])))
		self.data['resp_is'][nprev] = self.data['resp_vc'][nprev]
		self.data['resp_im'][nprev] = self.data['resp_is'][nprev] ** self.const['gamma_is']
		self.data['atten_i'][nprev] = self.const['a_is'] * self.data['resp_im'][nprev]
		

	def simulate(self):
		'''
		'''
		nstim = len(self.stimulus)

		response = np.zeros(nstim)      # speeds up matlab

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

				self.data['resp_ir'][ncurr] = self.data['gain_r'][nprev] * self.data['stim'][ncurr]
				
				self.data['resp_r'][ncurr] = (self.time_k['f1_tau_r'] * 
								self.data['resp_r'][nprev] + self.time_k['f2_tau_r'] * 
								self.data['resp_ir'][nprev] + self.time_k['f3_tau_r'] * 
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
				 
				self.data['resp_e'][ncurr] = (self.time_k['f1_tau_e'] * 
									self.data['resp_e'][nprev] + self.time_k['f2_tau_e'] * 
									self.data['resp_r'][nprev] + self.time_k['f3_tau_e'] * 
									self.data['resp_r'][ncurr])

				self.data['beta'][ncurr] = (self.const['c_beta'] + 
											self.const['rk_beta'] * self.data['resp_e'][ncurr])
				self.data['beta_e'][ncurr] = (self.data['beta'][ncurr] / 
											(1 + self.data['beta'][ncurr] / self.const['beta_e_max']))
				self.data['resp_q'][ncurr] = 1 / self.data['beta_e'][ncurr]

				# express tau_x in time steps
				self.data['tau_x'][ncurr] = self.data['resp_q'][ncurr] * self.const['nstep']    
				f1_tau_x = np.exp( - 1 / self.data['tau_x'][ncurr])
				f2_tau_x = self.data['tau_x'][ncurr] - (1 + self.data['tau_x'][ncurr]) * f1_tau_x
				f3_tau_x = 1 - self.data['tau_x'][ncurr] + self.data['tau_x'][ncurr] * f1_tau_x

				self.data['resp_x'][ncurr] = (f1_tau_x * self.data['resp_x'][nprev] + 
								  self.data['gain_x'][nprev] * f2_tau_x * self.data['resp_q'][nprev] + 
								  self.data['gain_x'][nprev] * f3_tau_x * self.data['resp_q'][ncurr])
				#print self.data['resp_x'], self.data['gain_x'][nprev]
				
				self.data['resp_os'][ncurr] = self.data['resp_x'][ncurr] ** self.const['rnx']
				           
				self.data['resp_c'][ncurr] = (self.time_k['f1_tau_c'] * self.data['resp_c'][nprev] +
									self.time_k['f2_tau_c'] * self.data['resp_os'][nprev] + 
								  	self.time_k['f3_tau_c'] * self.data['resp_os'][ncurr])
				 
				self.data['gain_x'][ncurr] = (1 / (1 + (self.const['a_c'] * 
												self.data['resp_c'][ncurr]) ** self.const['rnc']))
				   
				self.data['resp_vc'][ncurr] = self.data['resp_os'][ncurr] / self.data['atten_i'][nprev] 
				
				self.data['resp_is'][ncurr] = (self.time_k['f1_tau_vc'] * 
										self.data['resp_is'][nprev] + self.time_k['f2_tau_vc'] * 
										self.data['resp_vc'][nprev] + self.time_k['f3_tau_vc'] * 
										self.data['resp_vc'][ncurr])
				 
				self.data['resp_im'][ncurr] = self.data['resp_is'][ncurr] ** self.const['gamma_is']
				   
				self.data['atten_i'][ncurr] = (self.time_k['f1_tau_is'] * 
								self.data['atten_i'][nprev] + self.time_k['f2_tau_is'] * 
								self.data['resp_im'][nprev] + self.time_k['f3_tau_is'] * 
								self.data['resp_im'][ncurr])      
				
			resp_norm = ((self.data['resp_is'][ncurr] - self.dark['resp_is']) /
								self.dark['resp_is'] )
			response[istim] = resp_norm

		return self.stimulus, response

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
		 					self.const['tau_b0'] * self.const['rk_b'] * (stimn + 1) / stimn +
		 			     	np.sqrt((1 - self.const['rk_b'] - self.const['tau_r'] / 
		 			     	self.const['tau_b0'] * self.const['rk_b'] * 
		 			     	(stimn + 1) / stimn) ** 2 + 4 * self.const['rk_b'])))
		 	resp_r = (1 - resp_b) * stim / (1 + stimn)
		 	resp_e = resp_r

		beta = self.const['c_beta'] + self.const['rk_beta'] * resp_e
		beta_e = beta / (1 + beta / self.const['beta_e_max'])

		return (x  -  ((1  /  (1  +  (self.const['a_c']  *  x)  **  self.const['rnc'])) /
					beta_e)  **  self.const['rnx'])


def plot_cone():

	c = cone()
	stimulus, response = c.simulate()

	fig  = plt.figure(figsize=(10, 8))
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)

	ax1.semilogy(stimulus, 'k')

	ax2.plot(response, 'k')

	ax1.set_ylabel('luminance (td)')
	ax2.set_ylabel('normalized response')
	ax2.set_xlabel('time')

	plt.tight_layout()
	plt.show()

if __name__ == '__main__':

	plot_cone()