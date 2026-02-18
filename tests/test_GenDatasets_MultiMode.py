import os, sys
import pytest
import numpy as np
import perceval as pcvl
from perceval import *


SCRIPTPATH = os.path.abspath(__file__)
REPODIR = os.path.split(os.path.split(SCRIPTPATH)[0])[0]
sys.path.append(REPODIR)

from datasets import GenDatasets_MultiMode as Gen


@pytest.mark.parametrize("detector_type",['8Bin', 'PNR'])
def test_load_POVM(detector_type):
    POVM = Gen.load_POVM(detector_type, abs_path='./datasets')
    with pytest.raises(Exception):
        _ = Gen.load_POVM('TMD', abs_path='./datasets')

@pytest.fixture
def circuit():
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
    return decomposed_circuit

@pytest.mark.parametrize("state_species", ['coherent', 'squeezed','fock'])
def test_Perfect_Sampling(circuit, state_species):
    if state_species !='fock':
        initial_state = [0.6,0.4,0.2,0.5,0.0,0.0]
    else:
        initial_state = [1,2,1,1,1,2]
    s = Gen.Sample_from_random_unitary_6modes(circuit, state_species, initial_state, 10, 9)
    assert s.shape==(10,6)

def test_Sample_mixed_coherent_from_random_unitary_6modes(circuit):
    alpha0 = [0.6,0.4,0.2,0.5,0.0,0.0]
    alpha1 = [0.0,0.1,0.2,0.0,0.4,0.2]
    s = Gen.Sample_mixed_coherent_from_random_unitary_6modes(circuit, alpha0, alpha1, 10, 9)
    assert s.shape ==(10,6)

def test_Sample_lossy_fock_from_random_unitary_6modes(circuit):
    n0 = [3,3,3,3,3,3]
    s = Gen.Sample_lossy_fock_from_random_unitary_6modes(circuit, n0, 1000, 9)
    assert s.shape==(1000,6)

@pytest.mark.parametrize("state_species", ['coherent', 'squeezed','lossy_fock','mixed_coherent'])
def test_Sampling_with_POVM(circuit, state_species):
    kshots=10
    if state_species == 'mixed_coherent':
        initial_state = [[0.6,0.4,0.2,0.5,0.0,0.0],[0.0,0.1,0.2,0.0,0.4,0.2]]
    elif state_species not in ['fock','lossy_fock']:
        initial_state = [0.6,0.4,0.2,0.5,0.0,0.0]
    else:
        initial_state = [1,1,1,1,1,1]
        kshots=1000
    samples = Gen.Sampling_with_POVM('./datasets',circuit, state_species, initial_state, kshots, 9)
    assert samples.shape==(6,kshots)
