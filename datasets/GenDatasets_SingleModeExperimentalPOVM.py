import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import sys, os
from scipy import stats
from scipy import sparse
from scipy.special import factorial, binom

##############################################
#
# convert probability distr to click distr
#
##############################################

def ProbDistr1Mode(detector_type:str, species:str, ampl:float) -> np.array:
    if detector_type == '4Bin':
        cutoff = 189
    elif detector_type == 'SNSPD':
        cutoff = 443
    elif detector_type == '8Bin':
        cutoff = 158
    elif detector_type == 'PNR':
        cutoff = 729
    else:
        raise Exception('Please decide for either a \'4Bin\', \'8Bin\' or and \'SNSPD\' detector ')
    if species == 'Coherent':
        p = np.zeros(cutoff)
        nbar = ampl
        for c in range(cutoff):
            coeff = stats.poisson.pmf(c, mu=nbar)
            if coeff > 1e-15:
                p[c] = coeff
    elif species == 'Squeezed':
        p = np.zeros(cutoff)
        for c in range(0,cutoff,2):
            expr = binom(c,c/2)/np.cosh(ampl)*(np.tanh(ampl)/2)**c
            if expr >1e-15:
                p[c] = expr
    elif species == 'Thermal':
        p = np.array([(1/(1+ampl))*(ampl/(1+ampl))**c if (1/(1+ampl))*(ampl/(1+ampl))**c > 1e-15 else 0. for c in range(cutoff) ])
    elif species == 'PATS':
        p = np.array([c*(ampl**(c-1))/((1+ampl)**(c+1)) if c*(ampl**(c-1))/((1+ampl)**(c+1)) > 1e-15 else 0. for c in range(cutoff) ])
    else:
        raise Exception('Please decide for a species which is either \'Coherent\',\'Thermal\', \'Squeezed\' or \'PATS\'')
    return p

def ClickDistr(detector_type:str, species:str, ampl:float, num_samples:int, abs_path:str) -> (np.array, np.array):
    p = ProbDistr1Mode(detector_type, species, ampl)
    if detector_type =='SNSPD':
        POVM = np.load(f'{abs_path}/ExperimentalData/4-pixel SNSPD/POVM_4p.npy')
    elif detector_type == '4Bin':
        POVM = np.load(f'{abs_path}/ExperimentalData/4-bin spatial/POVM_4bin.npy')
    elif detector_type == '8Bin':
        POVM = np.load(f'{abs_path}/ExperimentalData/8-bin TMD/POVM_8bin.npy')
    elif detector_type == 'PNR':
        POVM = np.load(f'{abs_path}/ExperimentalData/PNR SNSPD/POVMs_EMG_4+_smoothing_1e-6_D40.npy')
    click_distr = POVM.T @ p
    # make sure that p sums up to 1
    if np.sum(click_distr[:-1]) < 1.:
        click_distr[-1] = 1 - np.sum(click_distr[:-1])
    else:
        click_distr[-1] = 0.0
        click_distr = click_distr/np.sum(click_distr)
        #click_distr = np.round(click_distr, 3)
        
    samples  = random.choice(np.arange(len(click_distr)), size=num_samples, p=click_distr)
    return click_distr, np.array(samples).reshape(1,1,num_samples)

##############################################
#
# Loading and processing experimentally
# measured coherent states
#
##############################################

def bins_to_int(a: np.array) -> np.array:
    len_a = np.shape(a)[0]
    a_in_ints = np.sum(a, axis=-1)
    return np.array(a_in_ints).reshape(1,len_a)

def load_exp_CoherentStates(detector_type:str, shots_per_state:int, abs_path:str) -> np.array:
    if detector_type == 'PNR':
        path = f'{abs_path}/ExperimentalData/PNR SNSPD/whichEventInfo/whichEventInfo'
        images = []
        for i in range(13):
            if i < 10:
                data = sparse.csr_matrix.todense(sparse.load_npz(f'{path}_0{i}.npz'))
            elif i >= 10:
                data = sparse.csr_matrix.todense(sparse.load_npz(f'{path}_{i}.npz'))
            images.append((data[:shots_per_state,:]).T)
        return np.array(images)
    elif detector_type == '8Bin':
        path = f'{abs_path}/ExperimentalData/8-bin TMD/whichBinInfo/whichBinInfo'
        images = []
        for i in range(13): # exclude last dataset
            if i < 10:
                raw_data = sparse.csr_matrix.todense(sparse.load_npz(f'{path}_0{i}.npz'))
                data_in_ints = bins_to_int(raw_data)
                images.append(data_in_ints[:,:shots_per_state])
            elif i >= 10:
                raw_data = sparse.csr_matrix.todense(sparse.load_npz(f'{path}_{i}.npz'))
                data_in_ints = bins_to_int(raw_data)
                images.append(data_in_ints[:,:shots_per_state])
        return np.array(images)

####################################
#
# Sampling of states
#
####################################

thermal_configs = np.zeros(14)
for l in range(1,15):
    thermal_configs[l-1]=0.5*l
    
squeezing_configs = np.zeros(12) 
for i in range(1,13):
    squeezing_configs[i-1]=round(0.1*i,2)
            
pats_config = np.zeros(10)
for idx, a1 in enumerate(range(5,15)):
    pats_config[idx] = 0.03*a1

def SampleClassicalStates(detector_type:str, shots_per_state:int, abs_path:str) -> dict:
    images = load_exp_CoherentStates(detector_type, shots_per_state, abs_path)
    print('Finished with loading experimental data')
    for t_config in thermal_configs:
        _, samples = ClickDistr(detector_type, 'Thermal', t_config, shots_per_state, abs_path)
        images = np.concatenate((images, samples))
    data = {'images': images, 'labels': np.zeros(np.shape(images)[0])}
    return data

def SampleNonClassicalStates(detector_type:str, shots_per_state:int, abs_path:str) -> dict:
    _, samples = ClickDistr(detector_type, 'Squeezed', 0.1, shots_per_state, abs_path)
    for ampl_squeezed in squeezing_configs[1:]:
        _, next_samples = ClickDistr(detector_type, 'Squeezed', ampl_squeezed, shots_per_state, abs_path)
        samples = np.concatenate((samples, next_samples))
    for ampl_pats in pats_config:
        _, next_samples = ClickDistr(detector_type, 'PATS', ampl_squeezed, shots_per_state, abs_path)
        samples = np.concatenate((samples, next_samples))
    data = {'images': samples, 'labels': np.ones(np.shape(samples)[0])}
    return data

##################################
#
# Using the code
#
##################################

if __name__ == '__main__':
    c_ds = SampleClassicalStates('8Bin', 90000, abs_path='.')
    nc_ds = SampleNonClassicalStates('8Bin', 90000, abs_path='.')
    SAVENAME = 'ds_test'
    np.savez(f'{SAVENAME}',cl_images=c_ds['images'], ncl_images=nc_ds['images'])
    #print(f'c_ds[\'images\'] has shape = {np.shape(c_ds["images"])} and labels have shape {np.shape(c_ds["labels"])}')
    #print(f'nc_ds[\'images\'] has shape = {np.shape(nc_ds["images"])} and labels have shape {np.shape(nc_ds["labels"])}')
