from __future__ import division
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pylab as plt

from base import spectsens as s
        

class cone_hc(object):    
    '''
    Fortran source code,  developed with an Intel Fortran90 compiler on Linux

    note: implicit type convention: variables starting with i..n are
    integers,  all others are reals

    9 Jan 2005,  author J.H. van Hateren
    Belongs to "A cellular and molecular model of response kinetics
    and adaptation in primate cones and horizontal cells",  J.Vision (2005)

    Remarks:
     - function rtbis not yet included (see below); insert that, 
     and change the statements for resp_c and resp_vs at 
     'set pre - stimulus values'; the values there now are only 
     correct for stim_prev = 100 td
    -  for higher gains in the feedback loops,  nrate may need to 
     be larger
    -  check the ARMA coefficients for very high or very small values
     of tau,  switch to double precision variables if necessary
    '''   
    def __init__(self, wvlen=559, peak_sens=559):
        '''
        '''
        self._gen_constants()
        self._gen_data_container()
        self._gen_time_constants()
        #self.get_dark_values()
        #sensitivity = s.neitz(LambdaMax=peak_sens, OpticalDensity=0.4, LOG=False,
        #            StartWavelength=wvlen, EndWavelength=wvlen,  
        #            resolution=1, EXTINCTION=False)
        self.gen_stimulus()
        #self.stimulus = self.stimulus * sensitivity
        self.get_adaptive_states()

    def _gen_data_container(self):
        '''
        '''
        self.data = {}
        self.data['stim']  =  np.zeros(2)                           #I in model
        self.data['resp_r']  =  np.zeros(2)                         #signal after tau_r
        self.data['resp_e']  =  np.zeros(2)                         #E *  in model
        self.data['beta']  =  np.zeros(2)                           #beta in model
        self.data['resp_q']  =  np.zeros(2)                         #Q in model
        self.data['tau_x'] =  np.zeros(2)                          #tau_x in model
        self.data['resp_w']  =  np.zeros(2)                         #alpha / beta in model
        self.data['gain_x']  =  np.zeros(2)                         #alpha in model
        self.data['resp_x']  =  np.zeros(2)                         #X in model
        self.data['resp_os']  =  np.zeros(2)                        #I_os in model
        self.data['resp_c']  =  np.zeros(2)                         #C in model
        self.data['resp_vc']  =  np.zeros(2)                        #signal after division by g_i
        self.data['resp_is']  =  np.zeros(2)                        #V_is in model
        self.data['gain_is']  =  np.zeros(2)                        #g_is in model
        self.data['gain_i']  =  np.zeros(2)                         #g_i in model
        self.data['resp_vs']  =  np.zeros(2)                        #V_s in model
        self.data['resp_ic']  =  np.zeros(2)                        #I_t in model
        self.data['resp_tr']  =  np.zeros(2)                        #signal after tau_1
        self.data['resp_ih']  =  np.zeros(2)                        #signal after tau_2 (V_b)
        self.data['resp_h']  =  np.zeros(2)                         #V_h in model
        self.data['resp_h0']  =  np.zeros(2)                        #V_is_prime in model
        self.data['stim_ar']  =  np.zeros(self.k['nstim_max'])                #stimulus array
        self.data['resp_ar']  =  np.zeros(self.k['nstim_max'])                #response array        
        self.data['resp_kl']  =  np.zeros(self.k['nstim_max'])                #auxiliary variable used for delay      

    def gen_stimulus(self):
        '''make stimulus
        '''

        ip1 = 25. * self.k['ratefrac']            #25 ms stimulus latency
        ip2 = ip1 + 100. * self.k['ratefrac'] - 1       #100 ms pulse width
        stim_dc = 100.                        #background illuminance 100 td
        self.data['stim_ar'][:self.k['nlen']] = stim_dc
        self.data['stim_ar'][ip1:ip2] = stim_dc + 2. * stim_dc     #contrast 2

    def _gen_constants(self):
        '''
        parameters,  in ms where applicable,  converted to delta_t by self.k['ratefrac']
        values are those of Figs.7 and 6A of the article
        '''
        self.k = {}
        self.k['nrate'] = 10000                ##timesteps  /  sec (delta_t = 100 us)
        self.k['ratefrac'] =  self.k['nrate']  /  1000. #conversion 1 ms to 100 us timebase
        self.k['nlen'] = int(300. * self.k['ratefrac']) #300 ms stimulus duration
        self.k['nstim_max'] = 5 * self.k['nrate']       #maximum stimulus length 5 sec
        self.k['stim_prev'] = 100.                      #assumed illuminance before start stimulus
        self.k['tau_r'] = 0.49 * self.k['ratefrac']     #parameter tau_r in model
        self.k['tau_e'] = 16.8 * self.k['ratefrac']     #parameter tau_e in model
        self.k['beta_0'] = 2.80e-3                      #parameter c_beta in model
        self.k['g_ex0'] = 1.63e-4                       #parameter k_beta in model
        self.k['rnx'] = 1.                              #parameter n_x in model
        self.k['rnc'] = 4.                              #parameter n_c in model
        self.k['tau_c'] = 2.89 * self.k['ratefrac']     #parameter tau_c in model
        self.k['a_c'] = 9.08e-2                         #parameter a_c in model
        self.k['tau_vc'] = 4. * self.k['ratefrac']      #parameter tau_m in model
        self.k['gamma_is'] = 0.678                      #parameter gamma_is in model
        self.k['tau_is'] = 56.9 * self.k['ratefrac']    #parameter tau_is in model
        self.k['a_is'] = 7.09e-2                        #parameter a_is in model
        self.k['ripmax'] = 151.1                        #parameter g_t in model
        self.k['vk'] =  - 10.                           #parameter v_k in model
        self.k['vn'] = 3.                               #parameter v_n in model
        self.k['tau_tr'] = 4. * self.k['ratefrac']      #parameter tau_1 in model 
        self.k['tau_ih'] = 4. * self.k['ratefrac']      #parameter tau_2 in model 
        self.k['tau_h'] = 20. * self.k['ratefrac']      #parameter tau_h in model 
        self.k['rdel'] = 2.82 * self.k['ratefrac']      #overall delay
        self.k['vh0'] = 19.7                            #parameter V_I in model 
        self.k['rho'] = 0.733                           #parameter mu in model 
        self.k['tau_h0'] = 250. * self.k['ratefrac']    #parameter tau_a in model 

    def _gen_time_constants(self):
        '''calculate ARMA coefficients of fixed filters
        '''

        self.time_k = {}
        self.time_k['f1_tau_r'] = np.exp( - 1. / self.k['tau_r'])
        self.time_k['f2_tau_r'] = (self.k['tau_r'] - (1.  +  self.k['tau_r']) * 
                self.time_k['f1_tau_r'])
        self.time_k['f3_tau_r'] = (1. - self.k['tau_r'] + self.k['tau_r'] * 
                self.time_k['f1_tau_r'])

        self.time_k['f1_tau_e'] = np.exp( - 1. / self.k['tau_e'])
        self.time_k['f2_tau_e'] = (self.k['tau_e'] - (1. + self.k['tau_e']) *
                self.time_k['f1_tau_e'])
        self.time_k['f3_tau_e'] = (1. - self.k['tau_e'] + self.k['tau_e'] * 
                self.time_k['f1_tau_e'])

        self.time_k['f1_tau_c'] = np.exp( - 1. / self.k['tau_c'])
        self.time_k['f2_tau_c'] = (self.k['tau_c'] - (1. + self.k['tau_c']) *
                self.time_k['f1_tau_c'])
        self.time_k['f3_tau_c'] = (1. - self.k['tau_c'] + self.k['tau_c'] * 
                self.time_k['f1_tau_c'])

        self.time_k['f1_tau_vc'] = np.exp( - 1. / self.k['tau_vc'])
        self.time_k['f2_tau_vc'] = (self.k['tau_vc'] - (1. + self.k['tau_vc']) * 
                self.time_k['f1_tau_vc'])
        self.time_k['f3_tau_vc'] = (1. - self.k['tau_vc'] + self.k['tau_vc'] * 
                self.time_k['f1_tau_vc'])

        self.time_k['f1_tau_is'] = np.exp( - 1. / self.k['tau_is'])
        self.time_k['f2_tau_is'] = (self.k['a_is'] * (self.k['tau_is'] - 
                    (1. + self.k['tau_is']) * self.time_k['f1_tau_is']))
        self.time_k['f3_tau_is'] = (self.k['a_is'] * (1. - self.k['tau_is'] + 
                    self.k['tau_is'] * self.time_k['f1_tau_is']))

        self.time_k['f1_tau_tr'] = np.exp( - 1. / self.k['tau_tr'])
        self.time_k['f2_tau_tr'] = (self.k['tau_tr'] - (1. + self.k['tau_tr']) * 
                    self.time_k['f1_tau_tr'])
        self.time_k['f3_tau_tr'] = (1. - self.k['tau_tr'] + self.k['tau_tr'] * 
                    self.time_k['f1_tau_tr'])

        self.time_k['f1_tau_h0'] = np.exp( - 1. / self.k['tau_h0'])
        self.time_k['f2_tau_h0'] = (self.k['tau_h0'] - (1. + self.k['tau_h0']) * 
                    self.time_k['f1_tau_h0'])
        self.time_k['f3_tau_h0'] = (1. - self.k['tau_h0'] + self.k['tau_h0'] * 
                    self.time_k['f1_tau_h0'])


    def get_adaptive_states(self):
        '''determine adaptive state
        '''
        nprev = 0
        self.data['resp_c'][nprev] = fsolve(self.steady, 0.)   #find steady - state C
        #self.data['resp_c'][nprev] = 14.12579           #delete if rtbis is available
        self.data['stim'][nprev] = self.k['stim_prev']
        self.data['resp_r'][nprev] = self.data['stim'][nprev]
        self.data['resp_e'][nprev] = self.data['resp_r'][nprev]
        self.data['beta'][nprev] = self.k['beta_0'] + self.k['g_ex0'] * self.data['resp_e'][nprev]
        self.data['resp_q'][nprev] = 1. / self.data['beta'][nprev]
        self.data['tau_x'][nprev] = self.data['resp_q'][nprev] * self.k['ratefrac']
        self.data['gain_x'][nprev] = 1. / (1. + (self.k['a_c'] * self.data['resp_c'][nprev]) ** self.k['rnc'])
        self.data['resp_w'][nprev] = self.data['gain_x'][nprev] * self.data['resp_q'][nprev]
        self.data['resp_x'][nprev] = self.data['gain_x'][nprev] * self.data['resp_q'][nprev]
        self.data['resp_os'][nprev] = self.data['resp_x'][nprev] ** self.k['rnx']
        self.data['resp_vc'][nprev] = (self.data['resp_os'][nprev] / self.k['a_is']) ** (1. / (1. + self.k['gamma_is']))                
        self.data['resp_is'][nprev] = self.data['resp_vc'][nprev]
        self.data['gain_is'][nprev] = self.data['resp_is'][nprev] ** self.k['gamma_is']
        self.data['gain_i'][nprev] = self.k['a_is'] * self.data['gain_is'][nprev]
        self.data['resp_h0'][nprev] = self.data['resp_is'][nprev]
        self.data['gtau'] = (self.data['resp_h0'][nprev] / self.k['vh0']) ** self.k['rho']                   #a_I in model
        self.data['gripmax'] = self.k['ripmax'] / self.data['gtau']
        self.data['st_is'] = self.data['resp_is'][nprev]                #transferred to steady_vs via common
        self.data['resp_vs'][nprev] = fsolve(self.steady_vs, -1.e3)  #find steady - state V_s
        #self.data['resp_vs'][nprev] =  - 12.92790            #delete if rtbis is available
        self.data['resp_ic'][nprev] = self.data['gripmax'] / (1. + np.exp( - (self.data['resp_vs'][nprev] - self.k['vk']) / self.k['vn']))
        self.data['resp_tr'][nprev] = self.data['resp_ic'][nprev]
        self.data['resp_ih'][nprev] = self.data['resp_tr'][nprev]
        self.data['resp_h'][nprev] = self.data['resp_ih'][nprev]

    def simulate(self):
        '''
        '''
        ncurr = 0
        nprev = 1
        for i in range(1, self.k['nlen']):                       #main loop

            #ncurr and nprev determine which element of,  e.g.,  self.data['resp_r'],  is
            #the current one,  self.data['resp_r'](1) or self.data['resp_r']  =  np.zeros(2)

            nkl = ncurr                        #swap values ncurr and nprev
            ncurr = nprev
            nprev = nkl

            self.data['stim'][ncurr] = self.data['stim_ar'][i] 

            self.data['resp_r'][ncurr] = (self.time_k['f1_tau_r'] * self.data['resp_r'][nprev] + self.time_k['f2_tau_r'] * 
                self.data['stim'][nprev] + self.time_k['f3_tau_r'] * self.data['stim'][ncurr])

            self.data['resp_e'][ncurr] = (self.time_k['f1_tau_e'] * self.data['resp_e'][nprev] + self.time_k['f2_tau_e'] * 
                self.data['resp_r'][nprev] + self.time_k['f3_tau_e'] * self.data['resp_r'][ncurr])
            self.data['beta'][ncurr] = self.k['beta_0'] + self.k['g_ex0'] * self.data['resp_e'][ncurr]
            self.data['resp_q'][ncurr] = 1. / self.data['beta'][ncurr]

            self.data['tau_x'][ncurr] = self.data['resp_q'][ncurr] * self.k['ratefrac']
            f1_tau_x = np.exp(-1. / self.data['tau_x'][ncurr])       #ARMA coefficients tau_x
            f2_tau_x = self.data['tau_x'][ncurr] - (1. + self.data['tau_x'][ncurr]) * f1_tau_x
            f3_tau_x = 1. - self.data['tau_x'][ncurr] + self.data['tau_x'][ncurr] * f1_tau_x

            self.data['resp_x'][ncurr] =(f1_tau_x * self.data['resp_x'][nprev] + self.data['gain_x'][nprev] * 
                f2_tau_x * self.data['resp_q'][nprev] + self.data['gain_x'][nprev] * 
                f3_tau_x * self.data['resp_q'][ncurr])

            self.data['resp_w'][ncurr] = self.data['gain_x'][nprev] * self.data['resp_q'][ncurr]  #not necessary,  only for
                                                        #figure

            self.data['resp_os'][ncurr] = self.data['resp_x'][ncurr] ** self.k['rnx']

            self.data['resp_c'][ncurr] = (self.time_k['f1_tau_c'] * self.data['resp_c'][nprev] + self.time_k['f2_tau_c'] * 
                self.data['resp_os'][nprev] + self.time_k['f3_tau_c'] * 
                self.data['resp_os'][ncurr])

            self.data['gain_x'][ncurr] = 1. / (1. + (self.k['a_c'] * self.data['resp_c'][ncurr]) ** self.k['rnc'])

            self.data['resp_vc'][ncurr] = self.data['resp_os'][ncurr] / self.data['gain_i'][nprev] 

            self.data['resp_is'][ncurr] = (self.time_k['f1_tau_vc'] * self.data['resp_is'][nprev] + 
                self.time_k['f2_tau_vc'] * self.data['resp_vc'][nprev] + 
                self.time_k['f3_tau_vc'] * self.data['resp_vc'][ncurr])

            self.data['gain_is'][ncurr] = self.data['resp_is'][ncurr] ** self.k['gamma_is']

            self.data['gain_i'][ncurr] = (self.time_k['f1_tau_is'] * self.data['gain_i'][nprev] + self.time_k['f2_tau_is'] * 
                    self.data['gain_is'][nprev] + self.time_k['f3_tau_is'] * self.data['gain_is'][ncurr])

            self.data['resp_h0'][ncurr] = (self.time_k['f1_tau_h0'] * self.data['resp_h0'][nprev] + self.time_k['f2_tau_h0'] * 
                self.data['resp_is'][nprev] + self.time_k['f3_tau_h0'] * self.data['resp_is'][ncurr])

            self.data['gtau'] = (self.data['resp_h0'][ncurr] / self.k['vh0']) ** self.k['rho']

            self.data['gripmax'] = self.k['ripmax'] / self.data['gtau']
            gtau_ih = self.k['tau_ih'] * self.data['gtau']                   #tau_2_prime in model
            gtau_h = self.k['tau_h'] * self.data['gtau']                     #tau_h_prime in model

            self.data['resp_vs'][ncurr] = self.data['resp_is'][ncurr] - self.data['resp_h'][nprev]

            self.data['resp_ic'][ncurr] = self.data['gripmax'] / (1. + np.exp( - (self.data['resp_vs'][ncurr] - self.k['vk']) / self.k['vn']))

            self.data['resp_tr'][ncurr] = (self.time_k['f1_tau_tr'] * self.data['resp_tr'][nprev] + self.time_k['f2_tau_tr'] * 
            self.data['resp_ic'][nprev] + self.time_k['f3_tau_tr'] * self.data['resp_ic'][ncurr])

            f1_tau_ih = np.exp( - 1. / gtau_ih)        #ARMA coefficients tau_2_prime
            f2_tau_ih = gtau_ih - (1. + gtau_ih) * f1_tau_ih
            f3_tau_ih = 1. - gtau_ih + gtau_ih * f1_tau_ih

            f1_tau_h = np.exp( - 1. / gtau_h)
            f2_tau_h = gtau_h - (1. + gtau_h) * f1_tau_h
            f3_tau_h = 1. - gtau_h + gtau_h * f1_tau_h

            self.data['resp_ih'][ncurr] = (f1_tau_ih * self.data['resp_ih'][nprev] + f2_tau_ih * 
            self.data['resp_tr'][nprev] + f3_tau_ih * self.data['resp_tr'][ncurr])
            self.data['resp_h'][ncurr] = (f1_tau_h * self.data['resp_h'][nprev] + f2_tau_h * 
            self.data['resp_ih'][nprev] + f3_tau_h * self.data['resp_ih'][ncurr])

            self.data['resp_ar'][i] = self.data['resp_h'][ncurr]   #output of resp_h (replace
                                                  #for obtaining other signals)

 
        #
        #  delay response by interpolation
        #
        for i in range(1,  self.k['nlen']):
            rel = i - self.k['rdel']
            if rel >= 0.:
                iel1 = rel
                iel2 = iel1 + 1
                d1 = rel - iel1
                d2 = 1. - d1
            else:
                iel1 = rel
                iel2 = iel1 - 1
                d1 = float(iel1) - rel
                d2 = 1. - d1

            if iel1 < 1: 
                iel1 = 1
            if iel1 > self.k['nlen']: 
                iel1 = self.k['nlen']
            if iel2 < 1: 
                iel2 = 1
            if iel2 > self.k['nlen']: 
                iel2 = self.k['nlen']
            self.data['resp_kl'][i] = d2 * self.data['resp_ar'][iel1]  + d1 * self.data['resp_ar'][iel2]

        self.data['resp_ar'][:self.k['nlen']] = self.data['resp_kl'][:self.k['nlen']]
        
        return self.data['stim_ar'],  self.data['resp_ar']

    def steady(self,  x):
    
        return x - (1. / (1. + (self.k['a_c'] * x) ** self.k['rnc']) / (self.k['beta_0'] + self.k['g_ex0'] * self.k['stim_prev'])) ** self.k['rnx']

    def steady_vs(self,  x):
    
        return x - (self.data['st_is'] - self.data['gripmax'] / (1. + np.exp( - (x - self.k['vk']) / self.k['vn'])))


def plot_cone():
    '''
    '''
    from base import plot as pf

    c = cone_hc()
    stimulus,  response = c.simulate()

    fig  = plt.figure()
    #ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(111)
    pf.AxisFormat()
    #pf.TufteAxis(ax1,  ['left'],  [5,  5])
    pf.TufteAxis(ax2,  ['left',  'bottom'],  [5,  5])

    #ax1.semilogy(stimulus,  'k')
    ax2.plot(response,  'k')
    ax2.plot((stimulus / np.max(stimulus)) * 31, 'k')

    #ax1.set_xlim([0, 2000])
    ax2.set_xlim([0, 3000])
    ax2.set_ylim([30, 40])
    #ax1.set_ylabel('luminance (td)')
    ax2.set_ylabel('normalized response')
    ax2.set_xlabel('time')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    plot_cone()