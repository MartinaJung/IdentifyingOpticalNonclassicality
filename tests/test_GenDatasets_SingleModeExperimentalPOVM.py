import pytest, os, sys
import numpy as np

SCRIPTPATH = os.path.abspath(__file__)
REPODIR = os.path.split(os.path.split(SCRIPTPATH)[0])[0]
sys.path.append(REPODIR)

from datasets import GenDatasets_SingleModeExperimentalPOVM as Gen

@pytest.mark.parametrize("detector_type",['4Bin', 'SNSPD','8Bin', 'PNR'])

def test_ProbDistr1Mode(detector_type):
    p_coh = Gen.ProbDistr1Mode(detector_type, 'Coherent', 0.1)
    p_sq = Gen.ProbDistr1Mode(detector_type, 'Squeezed', 0.1)
    p_th = Gen.ProbDistr1Mode(detector_type, 'Thermal', 1.1)
    p_pats = Gen.ProbDistr1Mode(detector_type, 'PATS', 0.1)
    assert np.isclose(np.sum(p_coh),1.)
    assert np.isclose(np.sum(p_sq),1.)
    assert np.isclose(np.sum(p_th),1.)
    assert np.isclose(np.sum(p_pats),1.)


    with pytest.raises(Exception, match='Please decide for a species which is either \'Coherent\',\'Thermal\', \'Squeezed\' or \'PATS\''):
        p = Gen.ProbDistr1Mode(detector_type, 'coherent', 0.1)
    with pytest.raises(Exception, match='Please decide for either a \'4Bin\', \'8Bin\' or and \'SNSPD\' detector '):
        p = Gen.ProbDistr1Mode('TMD', 'Coherent',0.1)

@pytest.mark.parametrize("detector_type",['4Bin', 'SNSPD','8Bin', 'PNR'])
def test_ClickDistr(detector_type):
    c_coh, samples_coh = Gen.ClickDistr(detector_type, 'Coherent', 0.1, 10, abs_path='./datasets')
    c_sq, samples_sq = Gen.ClickDistr(detector_type, 'Squeezed', 0.1, 10, abs_path='./datasets')
    c_pats, samples_pats = Gen.ClickDistr(detector_type, 'PATS', 0.1, 10, abs_path='./datasets')
    c_th, samples_th = Gen.ClickDistr(detector_type, 'Thermal', 0.1, 10, abs_path='./datasets')
    if detector_type in ['4Bin', 'PNR', 'SNSPD']:
        for c in [c_coh, c_sq, c_pats, c_th]:
            assert len(c)==5
            samples_pats.shape == (1,1,10)
    elif detector_type == '8Bin':
        for c in [c_coh, c_sq, c_pats, c_th]:
            assert len(c)==9
    assert np.isclose(np.sum(c_coh),1.)
    assert np.isclose(np.sum(c_sq),1.)
    assert np.isclose(np.sum(c_pats),1.)
    assert np.isclose(np.sum(c_th),1.)

@pytest.mark.parametrize("detector_type",['8Bin', 'PNR'])
def test_load_exp_CoherentStates(detector_type):
    a = Gen.load_exp_CoherentStates(detector_type, 10, abs_path='./datasets')
    assert np.array(a).shape==(13,1,10)

@pytest.mark.parametrize("detector_type",['8Bin', 'PNR'])
def test_Sampling(detector_type):
    s = Gen.SampleClassicalStates(detector_type, 10, './datasets')
    assert len(s['labels']) == 13+14
    s1 = Gen.SampleNonClassicalStates(detector_type, 10, abs_path='./datasets')
    assert len(s1['labels'])==12+10
