import os, sys
import pytest
import numpy as np

SCRIPTPATH = os.path.abspath(__file__)
REPODIR = os.path.split(os.path.split(SCRIPTPATH)[0])[0]
sys.path.append(REPODIR)

from datasets import GenDataset_SingleModePerfectPNRDetector as Gen

@pytest.mark.parametrize("state_species", ['coherent', 'squeezed', 'fock', 'thermal', 'pats'])
def test_Sampling_1mode(state_species):
    if state_species != 'fock':
        initial_state=[0.3]
    elif state_species=='fock':
        initial_state=[3]
    s = Gen.Sampling_1mode(state_species, initial_state, 1000, 29)
    assert s.shape==(1,1000)

def test_gen_ds():
    data = Gen.gen_ds(None,10,29)
    assert data['cl_images'].shape==(54,1,10)
    assert data['ncl_images'].shape==(32,1,10)