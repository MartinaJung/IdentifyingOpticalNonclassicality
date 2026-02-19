import os, sys
import pytest
import numpy as np

SCRIPTPATH = os.path.abspath(__file__)
REPODIR = os.path.split(os.path.split(SCRIPTPATH)[0])[0]
sys.path.append(REPODIR)

from PolynomialRegression import Regression_h5py as Reg

def test_load_EncoderCorrs():
    out = Reg.load_EncoderCorrs('./PolynomialRegression','ds14_PNR','10_4_1', num_encode_layers=1, regularization=1.0, kshots=100)
    lr,best_epoch, data_encoder_corrs, data_predictions, data_true_label = out
    assert best_epoch == 260
    assert data_predictions.shape == data_true_label.shape

def test_def_PolynomialRegression():
    RegressionObj = Reg.PolynomialRegression([1,1])
    assert hasattr(RegressionObj,'deg')
    assert callable(RegressionObj.fit)

@pytest.mark.parametrize('deg',[[0,1],[1,0],[1,1],[2,0],[0,2],[2,1]])
def test_Regression(deg):
    out = Reg.Regression('./PolynomialRegression',
                'ds14_PNR', 
                deg, 
                decoder_architecture='10_4_1', 
                num_encode_layers=1, 
                regularization=1.0, 
                kshots=100)
    score, c, inter, classical_accuracy, nonclassical_accuracy, lr, best_epoch=out
    assert len(c)==sum(deg)


def test_HierarchyOfRegression():
    deg=[[0,1],[1,0],[1,1],[2,0],[0,2],[2,1]]
    FILENAME='Testfile_PolynomialReg'
    out = Reg.HierarchyOfRegression('./PolynomialRegression',
                        FILENAME=FILENAME,
                        dataset='ds14_PNR',
                        polynomial_degrees=deg, 
                        decoder_architecture='10_4_1', 
                        num_encode_layers=1,
                        kshots=100)
    # Check that file was created
    for d in deg:
        deg_str='_'.join(str(di) for di in d)
        path=f'./PolynomialRegression/ResultsPolyRegression/ds14_PNR/1el/100kshots/{deg_str}'
        assert os.path.isfile(path+FILENAME)
        os.remove(path+FILENAME)

