import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from scipy.special import factorial
import time, sys, jax
from tqdm import tqdm
import jax.random
jax.config.update('jax_enable_x64', True)

import perceval as pcvl
from perceval import *
import perceval.components as comp
from perceval.algorithm import Sampler
key = jax.random.PRNGKey(seed=42)

U = jax.random.orthogonal(key,6, dtype='float64')
print(U)

##############################################
#
# Implement random unitary
#
##############################################

def create_optical_unitary():
    angles = [0.9445336227131641,0.554966085008221,
    1.2484260099250566,0.9279784616248398,
    1.4286538822168207,1.4861154769996658,
    0.3490191062759926,1.4033752429576103,
    1.2793483996919193,0.9775616616954067,
    1.0151609125875,1.0660317793620724,
    1.163393310052446,0.7864977009735303,
    1.2789624177400865]

    phases = [-0.0,-0.0,0.0,-0.0,3.141592653589793,-3.141592653589793,
    -3.141592653589793,-3.141592653589793,-3.141592653589793,-3.141592653589793,
    -3.141592653589793,-3.141592653589793,3.141592653589793,-1.7901478111201378e-17,
    -9.579025415681777e-17]

    modes = [(5, 6),(4, 5),(3, 4),(2, 3),(1, 2),(5, 6),(4, 5),(3, 4),
    (2, 3),(5, 6),(4, 5),(3, 4),(5, 6),(4, 5),(5, 6)]

    output_phases = [np.float64(2.457715126635385e-18), np.float64(-9.987320008426984e-17), 
    np.float64(3.141592653589793), np.float64(-1.2446420885833891e-16), 
    np.float64(-9.710591448881335e-17), np.float64(-2.760677275437882e-17)]

    decomposed_circuit = Circuit(6)
    for idx, (m1,m2) in enumerate(modes):
        decomposed_circuit.add((m1-1,m2-1), BS.Ry(theta=2*angles[idx], phi_tl=phases[idx]))
    for idx, p in enumerate(output_phases):
        decomposed_circuit.add(idx, PS(phi=p))

    print('resonstructed unitary', np.round(np.real(decomposed_circuit.compute_unitary()),5))
    print('error between U and reconstructed unitary:',np.linalg.norm(U-decomposed_circuit.compute_unitary()))
    return decomposed_circuit

########################################
#
# State configurations
#
########################################

coherent_configs = np.zeros((38,6))
for l in range(10):
    coherent_configs[l,:]=0.1*l
for l1 in range(1,15):
    coherent_configs[9+l1,:3] = 0.1*l1
    coherent_configs[23+l1,3:] = 0.1*l1

squeezing_configs = np.zeros((22,6))
for i in range(1,7):
    squeezing_configs[i-1,:]=round(0.1*i,2)
for i1 in range(1,9):
    squeezing_configs[5+i1,:3]=round(0.1*i1,2)
    squeezing_configs[13+i1,3:]=round(0.1*i1,2)

fock_configs = np.zeros((15,6))
for h in range(1,6):
    fock_configs[h-1,:] = h
    for h1 in range(1,6):
        fock_configs[4+h1,:3]=h1
        fock_configs[9+h1,3:]=h1


#########################################
#
# Fock basis coefficients
#
#########################################

def coeff_coh(alpha:float) -> np.array:
   alp = round(alpha,2)
   return np.array([np.exp(-0.5*alp**2)*(alp**l)/(np.sqrt(factorial(l))) for l in range(30)])   
def coeff_squeezed(r:float) -> np.array:
   r = round(r,2)
   return np.array([1/(np.sqrt(np.cosh(r))*2**(l)*factorial(l))*(np.sqrt(factorial(2*l))*np.tanh(r)**l) for l in range(19)])

#########################################
#
# Load detection POVM
#
#########################################

def load_POVM(detector_type:str, abs_path:str) -> np.array:
    path = f'{abs_path}/ExperimentalData/'
    if detector_type =='SNSPD':
        POVM = np.load(path+'4-pixel SNSPD/POVM_4p.npy')
    elif detector_type == '4Bin':
        POVM = np.load(path+'4-bin spatial/POVM_4bin.npy')
    elif detector_type == '8Bin':
        POVM = np.load(path+'8-bin TMD/POVM_8bin.npy')
    elif detector_type == 'PNR':
        POVM = np.load(path+'PNR SNSPD/POVMs_EMG_4+_smoothing_1e-6_D40.npy')
    else:
        raise Exception('Please choose one of the following options: [SNSPD, 4Bin, 8Bin, PNR]')
    return POVM

# turn (single) sample from perfect PNR detector into sample from realistic detector
def ClickDistr(POVM:np.array, sample_from_unitary: np.array) -> np.array:
    samples = []
    for n in sample_from_unitary:
        p = np.zeros(POVM.shape[0])
        p[int(n)] = 1.
        click_distr = POVM.T @ p
        # make sure that p sums up to 1
        if np.sum(click_distr[:-1]) < 1.:
            click_distr[-1] = 1 - np.sum(click_distr[:-1])
        else:
            click_distr[-1] = 0.0
            click_distr = click_distr/np.sum(click_distr)
            #click_distr = np.round(click_distr, 3)
        samples.append(random.choice(np.arange(len(click_distr)), size=1, p=click_distr))
    return np.array(samples).reshape(1,len(sample_from_unitary))

##########################################
#     
# Sample (perfect) pure states
#
##########################################

def Sample_from_random_unitary_6modes(circuit, state_species:str, initial_state: list, shots:int, cutoff:int) -> np.array:
    '''         
        state_species (string): Defines the species of state to be simulated
            Allowed values are 'coherent', 'squeezed', 'thermal'
         
        initial_state (list): Specifies the initial state in the form [alpha_i]_i
            or [ri]_i
         
        shots (integer): Defines the number of measurement shots per state
    '''
    # create state vector
    sv = StateVector()
    M = len(initial_state)
   
    if state_species == 'squeezed':
        c = []
        for r in initial_state:
            c.append(coeff_squeezed(r))
        c = np.array(c) # shape(num_modes, cutoff/2)
        CUTOFF = int(cutoff/2)

        for m0 in tqdm(range(CUTOFF)):
            for m1 in range(CUTOFF):
                for m2 in range(CUTOFF):
                    for m3 in range(CUTOFF):
                        for m4 in range(CUTOFF):
                            for m5 in range(CUTOFF):
                                m = np.array([m0,m1,m2,m3,m4,m5])
                                coeff = [c[i,mi] for (i,mi) in zip(np.arange(6),m)]
                                coeff = np.prod(np.array(coeff))
                                if coeff >= 1e-4:
                                    sv += float(coeff)*StateVector(2*m) 
    elif state_species == 'coherent':
        c = []
        for a in initial_state:
            c.append(coeff_coh(a))
        c = np.array(c)
        CUTOFF = cutoff
        for m0 in tqdm(range(CUTOFF)):
            for m1 in range(CUTOFF):
                for m2 in range(CUTOFF):
                    for m3 in range(CUTOFF):
                        for m4 in range(CUTOFF):
                            for m5 in range(CUTOFF):
                                m = np.array([m0,m1,m2,m3,m4,m5])
                                coeff = [c[i,mi] for (i,mi) in zip(np.arange(6),m)]
                                coeff = np.prod(np.array(coeff))
                                if coeff >= 1e-4:
                                    sv += float(coeff)*StateVector(m) 
    
    elif state_species == 'fock':
        c = []
        sv += StateVector(np.array(initial_state,dtype='i'))

    # Sampling
    if state_species in ['coherent', 'squeezed', 'fock']:
        if not np.any(initial_state):
            results = np.zeros((shots,M))
        else:
            proc = pcvl.Processor("SLOS", circuit)
            #for mi in range(6):
            #    proc.add(mi, pcvl.LC(.1))
            proc.min_detected_photons_filter(0)
            proc.with_input(sv)
            sampler = pcvl.algorithm.Sampler(proc)
            samples = sampler.samples(shots)
            results = np.array(samples['results'])
            # Sample zeros by hand
            coeff0 = sv[BasicState([0]*M )]
            num_zeros =int( (np.abs(coeff0)**2)*shots )
            if num_zeros > 0:
                results = np.concatenate( (results[:-num_zeros], np.zeros((num_zeros,M))), axis=0 )
        return results

##########################################
#     
# Sample (perfect) mixed states
#
##########################################

def Sample_mixed_coherent_from_random_unitary_6modes(circuit, alpha0:float, alpha1:float, shots:int, cutoff:int) -> np.array:
    results0 = Sample_from_random_unitary_6modes(circuit,'coherent', alpha0, shots, cutoff)
    print('results0 have shape', results0.shape)
    results1 = Sample_from_random_unitary_6modes(circuit,'coherent', alpha1, shots, cutoff)
    results = np.concatenate((results0[:(shots//2),:], results1[:(shots//2),:]), axis=0)
    np.random.shuffle(results)
    return results

def Sample_lossy_fock_from_random_unitary_6modes(circuit, n0:list, shots:int, cutoff:int) -> np.array:
    results = Sample_from_random_unitary_6modes(circuit,'fock', n0, int(0.9*shots), cutoff)
    print('results have shape', results.shape)
    # single-photon loss, there is 10% loss in total
    lossy_fock_states = []
    for i in range(6):
        tmp = np.array([n for n in n0])
        tmp[i] = max(0,n0[i]-1)
        print(tmp)
        lossy_fock_states.append(tmp)
        lossy_results = Sample_from_random_unitary_6modes(circuit,'fock', tmp, int(0.017*shots), cutoff)
        print('lossy results have shape', lossy_results.shape)
        results = np.concatenate((results, lossy_results))
    print('shape of results:', results.shape)
    np.random.shuffle(results)
    return results[:shots,:]

##########################################
#     
# Sample truncated statistics
#
##########################################

def Sampling_with_POVM(abs_path:str, circuit, state_species:str, initial_state:list, shots:int, cutoff:int) -> np.array:
    POVM = load_POVM('PNR',abs_path)
    if state_species in ['coherent', 'squeezed', 'fock']:
        start = time.time()
        tmp = Sample_from_random_unitary_6modes(circuit, state_species, initial_state, shots, cutoff)
        end = time.time()
        print('Sampling with perceval took', end-start)
        samples = []
        start = time.time()
        for r in tmp:
            samples.append(ClickDistr(POVM, r))
        measurements = np.array(samples).reshape(tmp.shape)
        end = time.time()
        print('Turning the PNR distribution into a click dis. took', end-start)
        images =  np.array(measurements).reshape(len(initial_state), shots)
    
    elif state_species == 'mixed_coherent':
        #for initial_state in initial_states:
        tmp = Sample_mixed_coherent_from_random_unitary_6modes(circuit, 
                                                                initial_state[0], 
                                                                initial_state[1], 
                                                                shots, 
                                                                cutoff)
        samples = []
        for r in tmp:
            samples.append(ClickDistr(POVM, r))
        measurements = np.array(samples).reshape(tmp.shape)
        images =  np.array(measurements).reshape(-1, shots)

    elif state_species == 'lossy_fock':
        tmp = Sample_lossy_fock_from_random_unitary_6modes(circuit,
                                                                initial_state,
                                                                shots,
                                                                cutoff)
        samples = []
        for r in tmp:
            samples.append(ClickDistr(POVM, r))
        measurements = np.array(samples).reshape(tmp.shape)
        images =  np.array(measurements).reshape(len(initial_state), shots)
    return images

##################################
# 
# Use code
#
##################################

if __name__ == '__main__':
    STATE_SPECIES= sys.argv[1]
    KSHOTS = int(sys.argv[2])
    IMG_IDX = int(sys.argv[3])
    start_total = time.time()
    if STATE_SPECIES == 'coherent':
        initial_states = coherent_configs
    elif STATE_SPECIES == 'squeezed':
        initial_states = squeezing_configs
    elif STATE_SPECIES == 'fock':
        initial_states = fock_configs
    elif STATE_SPECIES == 'lossy_fock':
        initial_states = fock_configs
    decomposed_circuit = create_optical_unitary()
    #samples = Sampling_with_POVM('.',decomposed_circuit,STATE_SPECIES, [0,0,0,2,2,1], KSHOTS*1000, 24)
    samples = Sample_lossy_fock_from_random_unitary_6modes(decomposed_circuit,[0,0,0,2,2,1], KSHOTS*1000, 9)
    end_total = time.time()
    print('elapsed time:', end_total-start_total)
    #print(samples[:,:100])
    #np.savez(f'ds16_6modes_{STATE_SPECIES}_{IMG_IDX}_train_{KSHOTS}kshots', images=samples)