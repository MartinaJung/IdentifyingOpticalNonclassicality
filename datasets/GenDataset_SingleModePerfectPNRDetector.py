import numpy as np
import math, time
from scipy import stats
from scipy.special import factorial
import matplotlib.pyplot as plt

import perceval as pcvl
from perceval import *
from perceval.algorithm import Sampler
from tqdm import tqdm
np.random.seed(42)

##################################
#
# Fock basis coefficients
#
##################################

#def coeff_cat(alpha):
#   alp = round(alpha,2)
#   return np.array([np.sqrt(2/(1+np.exp(-2*(alp**2))*factorial(2*l)) )*(alp**(2*l)) for l in range(19)])

def coeff_coh(alpha):
   alp = round(alpha,2)
   return np.array([np.exp(-0.5*alp**2)*(alp**l)/(np.sqrt(factorial(l))) for l in range(29)])

def coeff_thermal(m, n_bar):
   return ((n_bar)**m)/((1+n_bar)**(m+1))
   
def coeff_squeezed(r):
   r = round(r,2)
   return np.array([1/(np.sqrt(np.cosh(r))*2**(l)*factorial(l))*(np.sqrt(factorial(2*l))*np.tanh(r)**l) for l in range(19)])

def coeff_pats(m, n_bar):
   if m == 0:
      return 0
   else:
      return m*(n_bar**(m-1))/((1+n_bar)**(m+1))

##############################
# 
# state configurations
#
##############################

squeezing_configs = []
for i in range(1,13):
   squeezing_configs.append([round(0.1*i,2)])
            
photon_added_thermal_config = [] # 0.25 - 1.2
for a1 in range(5,25):
   photon_added_thermal_config.append([0.05*a1])
            
coherent_configs = [] # 0 - 3.5
for l in range(36):
   coherent_configs.append([0.1*l])

mixed_coherent_configs = [] # should be of the form (FLAGS.beam_splitter,[[config1],[config2]], FLAGS.shots)
for idx, c in enumerate(coherent_configs[:-18]):
   mixed_coherent_configs.append([c,coherent_configs[idx+18]])

#############################
#
# Single-mode measurement
#
#############################

def Sampling_1mode(state_species, initial_state, shots, cutoff):
   #TODO: This function is an overkill 
   # --> write function to extract vector of diagonal entries to sample from
   ''' 
      beamsplitter (boolean): Dummy input to make the function look similar to the 3 mode version
      
      state_species (string): Defines the species of state to be simulated
         Allowed values are 'fock', 'coherent','squeezed', 'thermal', 'pats'
         
      initial_state (list): Specifies the initial state in the form [alpha0, alpha1, alpha2]
         or [r0, r1, r2]
         
      shots (integer): Defines the number of measurement shots per state
   '''   
   sv = StateVector()
   M = len(initial_state)
   
   if state_species == 'fock':
      sv += StateVector(initial_state)
            
   elif state_species == 'squeezed':
      c = []
      for r in initial_state:
         c.append(coeff_squeezed(r))
      c = np.array(c)
      for n in range(int(cutoff/2)):
         if c[0,n] >= 1e-3:
            coeff = c[0,n]
            sv += float(coeff)*StateVector([2*n]) #
                                 
   elif state_species == 'coherent':
      c = []
      for a in initial_state:
         nbar = a**2
         samples = np.random.poisson(lam=nbar, size=shots)
         samples = np.clip(samples, a_min=0, a_max = cutoff)
      return np.array(samples).reshape(1,shots)
                    
   elif state_species == 'thermal':
      a = []
      # entries of the Pats density matrix
      itr = pcvl.utils.states.max_photon_state_iterator(M,cutoff)
      for idx, config in enumerate(itr):
         tmp = 1
         for mode in range(M):
            tmp *= coeff_thermal(config[mode],initial_state[mode])
         a.append(tmp)
      input_dm = DensityMatrix(np.diag(a), m=M, n_max=cutoff)
      
   elif state_species == 'pats':
      a = []
      itr = pcvl.utils.states.max_photon_state_iterator(M,cutoff)
      for idx, config in enumerate(itr):
         tmp = 1
         for mode in range(M):
            tmp *= coeff_pats(config[mode],initial_state[mode])
         a.append(tmp)
      input_dm = DensityMatrix(np.diag(a), m=M, n_max=cutoff)
      
   # create circuit
   my_circuit = Circuit(M)
   
   # Sampling
   if state_species in ['squeezed','fock']:
      if initial_state == [0.0]*M:
         results = np.zeros((shots,M))
      else:
         proc = pcvl.Processor("SLOS", my_circuit)
         proc.min_detected_photons_filter(0)
         proc.with_input(sv)
         sampler = pcvl.algorithm.Sampler(proc)
         samples = sampler.samples(shots)
         results = np.array(samples['results'])
      
         # Sample zeros by hand
         coeff0 = sv[BasicState([0]*M )]
         num_zeros =int( (np.abs(coeff0)**2)*shots )
         if num_zeros > 0:
            results = np.concatenate( (results[:-num_zeros], np.zeros((num_zeros,1))), axis=0 )
      return results.reshape(1,shots)
   
   elif state_species=='thermal' or state_species=='pats':
      bell_simulator = Simulator(SLOSBackend())
      bell_simulator.set_circuit(my_circuit)
      bell_out_dm = bell_simulator.evolve_density_matrix(input_dm)
      results = np.array(bell_out_dm.sample(shots))
      return results.reshape(1,shots)


def Sampling_MixedCoherent(initial_states, shots, cutoff):
   measurements = []
   for state in initial_states:
      measurements.extend(Sampling_1mode('coherent', state, shots, cutoff).flatten())
   np.random.shuffle(measurements)
   print('shape of measurements is', np.array(measurements).shape)
   return (np.array(measurements)[:shots]).reshape(1,shots)

#####################################
#
# function for generating datasets
#
#####################################

def gen_ds(name_dataset, shots, cutoff):
   img_classical = []
   img_non_classical = []
   fails = 0
   fail_configs = []
   for config in squeezing_configs:
      try:
         img_non_classical.append(Sampling_1mode('squeezed',config, shots, cutoff))
      except:
         fails += 1
         fail_configs.append(config)
   for config in photon_added_thermal_config:
      img_non_classical.append(Sampling_1mode('pats',config, shots, cutoff))

   for config in tqdm(coherent_configs):
      img_classical.append(Sampling_1mode('coherent',config, shots, cutoff))
   for config in tqdm(mixed_coherent_configs):
      img_classical.append(Sampling_MixedCoherent(config, shots, cutoff))

   # save data in npz file
   if name_dataset != None:
      np.savez(f'{name_dataset}_train_{shots}shots', cl_images=img_classical, ncl_images=img_non_classical)
   return {'cl_images': np.array(img_classical), 'ncl_images': np.array(img_non_classical)}



if __name__ == "__main__":
   start = time.perf_counter()
   s = Sampling_1mode(False, 'pats', [0.2], shots=100000, cutoff=29)
   fig, ax = plt.subplots()
   #print(s)
   #s = BS50Mixed(True, [[3.1,3.1],[0.0,0]], shots=100, cutoff=20)
   ax.hist(s, bins = np.arange(np.max(s)+1))
   end = time.perf_counter()
   plt.show()
   print(f'elapsed time is {end-start}')