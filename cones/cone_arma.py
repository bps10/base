#! /usr/bin/env python
from __future__ import division
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pylab as plt

import os
import sys
sys.path.append(os.getcwd())

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
		# abbreviate for speed:
		_dat = self.data
		_k = self.k
		_stim = self.stimulus
		_const = self.const
		exp = np.exp

		nstim = len(_stim)
		response = np.zeros(nstim)

		for istim in range(0, nstim):
			for istep in range(0, _const['nstep']):
			   
				if istim == 0 and istep == 0:
					ncurr = 0             # current element
					nprev = 1             # previous element
				else:
					nkl = ncurr           # swap ncurr / nprev
					ncurr = nprev
					nprev = nkl

				_dat['stim'][ncurr] = _stim[istim] 

				_dat['resp_ir'][ncurr] = (_dat['gain_r'][nprev] * 
							_dat['stim'][ncurr])
				
				_dat['resp_r'][ncurr] = (_k['f1_tau_r'] * 
							_dat['resp_r'][nprev] + _k['f2_tau_r'] * 
							_dat['resp_ir'][nprev] + _k['f3_tau_r'] * 
							_dat['resp_ir'][ncurr])
				
				tau_b = _const['tau_b0'] * (_dat['resp_b'][nprev] + 
							_const['rk_b']) / _const['rk_b']
				
				gain_b = _const['cn'] * tau_b / _const['tau_r']
				
				f1_tau_b = exp( - 1 / tau_b)
				f2_tau_b = gain_b * (tau_b - (1 + tau_b) * f1_tau_b)
				f3_tau_b = gain_b * (1 - tau_b + tau_b * f1_tau_b)
				
				_dat['resp_b'][ncurr] = (f1_tau_b * _dat['resp_b'][nprev] + 
							f2_tau_b * _dat['resp_r'][nprev] + 
							f3_tau_b * _dat['resp_r'][ncurr])
				
				_dat['gain_r'][ncurr] = (1 - _dat['resp_b'][ncurr] -
							_const['cn'] * _dat['resp_r'][ncurr])
				 
				_dat['resp_e'][ncurr] = (_k['f1_tau_e'] * 
							_dat['resp_e'][nprev] + _k['f2_tau_e'] * 
							_dat['resp_r'][nprev] + _k['f3_tau_e'] * 
							_dat['resp_r'][ncurr])

				_dat['beta'][ncurr] = (_const['c_beta'] + 
							_const['rk_beta'] * _dat['resp_e'][ncurr])
				_dat['beta_e'][ncurr] = (_dat['beta'][ncurr] / 
							(1 + _dat['beta'][ncurr] / _const['beta_e_max']))
				_dat['resp_q'][ncurr] = 1 / _dat['beta_e'][ncurr]

				# express tau_x in time steps
				_dat['tau_x'][ncurr] = (_dat['resp_q'][ncurr] * 
							_const['nstep'])
				f1_tau_x = exp( - 1 / _dat['tau_x'][ncurr])
				f2_tau_x = (_dat['tau_x'][ncurr] - 
							(1 + _dat['tau_x'][ncurr]) * f1_tau_x)
				f3_tau_x = (1 - _dat['tau_x'][ncurr] + 
							_dat['tau_x'][ncurr] * f1_tau_x)

				_dat['resp_x'][ncurr] = (f1_tau_x * _dat['resp_x'][nprev] + 
							_dat['gain_x'][nprev] * f2_tau_x * 
							_dat['resp_q'][nprev] + _dat['gain_x'][nprev] * 
							f3_tau_x * _dat['resp_q'][ncurr])

				
				_dat['resp_os'][ncurr] = (_dat['resp_x'][ncurr] ** 
							_const['rnx'])
				          
				_dat['resp_c'][ncurr] = (_k['f1_tau_c'] * 
							_dat['resp_c'][nprev] + _k['f2_tau_c'] * 
							_dat['resp_os'][nprev] + _k['f3_tau_c'] * 
							_dat['resp_os'][ncurr])
				 
				_dat['gain_x'][ncurr] = (1 / (1 + (_const['a_c'] * 
							_dat['resp_c'][ncurr]) ** _const['rnc']))
				   
				_dat['resp_vc'][ncurr] = (_dat['resp_os'][ncurr] / 
							_dat['atten_i'][nprev] )
				
				_dat['resp_is'][ncurr] = (_k['f1_tau_vc'] * 
							_dat['resp_is'][nprev] + _k['f2_tau_vc'] * 
							_dat['resp_vc'][nprev] + _k['f3_tau_vc'] * 
							_dat['resp_vc'][ncurr])
				 
				_dat['resp_im'][ncurr] = (_dat['resp_is'][ncurr] ** 
							_const['gamma_is'])
				   
				_dat['atten_i'][ncurr] = (_k['f1_tau_is'] * 
							_dat['atten_i'][nprev] + _k['f2_tau_is'] * 
							_dat['resp_im'][nprev] + _k['f3_tau_is'] * 
							_dat['resp_im'][ncurr])      
				
			resp_norm = ((_dat['resp_is'][ncurr] - self.dark['resp_is']) /
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
	fig.set_tight_layout(True)
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
	fig.set_tight_layout(True)
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

	plt.show()


x = np.arange(0, np.pi * 2, 0.005)
background_illum=2000
contrast = 100
k = background_illum * (contrast / 100.)


def _parallel(freq):
	'''
	'''
	c = cone()
	sinewave = (k * np.sin(x * freq)) + background_illum
	c.set_stimulus(sinewave)
	response = c.simulate()
	return response


def temporal_sensitivity(background_illum=1000, contrast=100, 
	parallel=True):
	'''
	'''

	if parallel:
		from multiprocessing import Pool, cpu_count
		pool = Pool(processes=cpu_count())
		freqs = 10 ** np.arange(0, 3, 0.1)
		results = pool.map(_parallel, freqs)

		amp = np.zeros(len(freqs))
		for i, result in enumerate(results):
			amp[i] = abs(np.max(result) - np.min(result))

	else:
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

	return freqs, amp

def plot_temp_sens():
	'''
	'''
	freqs, amp = temporal_sensitivity()

	fig  = plt.figure()
	fig.set_tight_layout(True)
	ax1 = fig.add_subplot(111)

	pf.AxisFormat()
	pf.TufteAxis(ax1, ['left', 'bottom'], [5, 5])

	ax1.semilogx(freqs, amp / np.max(amp), 'ko-')

	ax1.set_xlabel('frequency (Hz)')
	ax1.set_ylabel('sensitivity')

	plt.show()


if __name__ == '__main__':

	from base import plot as pf
	#plot_temp_sens()
	#plot_frequencies()
	plot_cone()
	
