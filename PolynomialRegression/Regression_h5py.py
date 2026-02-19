import jax.numpy as jnp
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Callable
import pathlib, sys, os, glob, h5py
from tqdm import tqdm

def load_EncoderCorrs(abs_path:str,
                    dataset:str, 
                    decoder_architecture:str, 
                    num_encode_layers:int, 
                    regularization:float, 
                    kshots:int) -> (float,float,np.array,np.array,np.array):
    ''' 
        Loads tuples (enocer_outputs, labels) that are fitted with the polynomials
        input:
            decoder_architecture: string like _14_4_2 describing architecture of dense decoder
            num_encode_layers: number of encoding layers
            regularization: regularization strength
            kshots: number of samples per state in [k]
    '''
    SEARCH_PATH = f"{abs_path}/EncoderOutputsAndModelsPrediction/" + dataset + f"/{num_encode_layers}el/" \
             + f"{kshots}kshots"
    regu_str = np.format_float_positional(regularization,unique=False, precision=1)
    FILENAME = glob.glob(f'{SEARCH_PATH}/*RGZN{regu_str}*lr0.01*{decoder_architecture}.hdf5')
    try:
        FILENAME = FILENAME[0]
        with h5py.File(FILENAME, "r") as fp:
            data_encoder_corrs = fp['encoder_corrs']
            data_predictions = fp['predictions']
            data_true_labels = fp['true_labels']
            best_epoch = fp['best_epoch']
            lr = fp.attrs['LEARNING_RATE']
            return lr, np.array(best_epoch), np.array(data_encoder_corrs), np.array(data_predictions), np.array(data_true_labels)
    except:
        raise Exception('I could not find a file. Please check your file description.')

################################################
#
# defining an instance of Polynomial Regression
#
################################################

class PolynomialRegression:
    def __init__(self, deg):
        self.deg = deg
        LinearRegression()
    def fit(self, x,y) -> (np.array, float, Callable, float):
        X = []
        for i,d in enumerate(self.deg):
            X.extend([x[:,i]**di for di in range(1,d+1)])
        X = jnp.array(X).T
        reg = LinearRegression().fit(X, y)
        predictions = reg.predict(X)
        return reg, np.round(reg.score(X, y),5), predictions

def Regression(abs_path:str,
                dataset:str, 
                deg:list, 
                decoder_architecture:str, 
                num_encode_layers:int, 
                regularization:float, 
                kshots:int) -> (float, np.array, float, float, float, float, int):
    '''
        Perfoms a polynomial regression for a certain degree

        input:
            deg: list [e0,e1,..,eL] describing the maximal order up to which an encoder output enters the
                polynomial. For example, [1,2] corresponds to a polynomial f=(x1) + (x2) + (x2)^2 + b
            decoder_architecture: string like _14_4_2_ describing the dense decoder's architecture
            num_encode_layers: number of encoding layers
            regularization: regularization strength
            kshots: number of samples per states in [k]
    '''
    lr, best_epoch, encoder_corrs, predictions, true_labels = load_EncoderCorrs(abs_path,
                                            dataset,
                                            decoder_architecture, 
                                            num_encode_layers, 
                                            regularization,
                                            kshots)
    RegressionObj = PolynomialRegression(deg)
    reg, score, poly_predictions = RegressionObj.fit(encoder_corrs, predictions)
    c, inter, = reg.coef_, reg.intercept_
 
    # check accuracy of approximating polynomial
    len_nonclassical_states = sum(true_labels)
    len_classical_states = len(true_labels)-len_nonclassical_states
    classical_accuracy = 0
    nonclassical_accuracy = 0
    for idx, t in enumerate(true_labels):
        if t == 0:
            classical_accuracy += int(0<poly_predictions[idx])
        elif t==1:
            nonclassical_accuracy += int(0>poly_predictions[idx])
    classical_accuracy = classical_accuracy/len_classical_states
    nonclassical_accuracy = nonclassical_accuracy/len_nonclassical_states
    total_accuracy = np.mean(poly_predictions == true_labels) 
    return score, c, inter, classical_accuracy, nonclassical_accuracy, lr, best_epoch

# Hierarchy of Polynomial Regression with increasing order of polynoms
def HierarchyOfRegression(abs_path:str,
                        FILENAME:str,
                        dataset:str,
                        polynomial_degrees:list, 
                        decoder_architecture: str, 
                        num_encode_layers:int,
                        kshots:int):
    '''
        Performs a polynomial regression of increasing order for different regularization strengths
        
        input:
            polynomial_degrees: list of lists [e0,e1,...,e_L]. Each element e_i represents 
                                the maximal exponential order of which
            decoder_architecture: string like _14_4_2_ describing the dense decoder's architecture
            num_encode_layers: number of encoding layers
            kshots: number of samples per states in [k]
    '''
    for polynomial_degree in polynomial_degrees:
        print(f'polynomial order: {polynomial_degree}')
        score, coefficients, inter = [], [], []
        cl_accuracy, ncl_accuracy=[],[]
        best_epochs =[]
        for regu in tqdm(range(31)):
            r = 0.1*regu
            Regression_out = Regression(abs_path,dataset,polynomial_degree, decoder_architecture, num_encode_layers, r, kshots)
            score.append(Regression_out[0])
            coefficients.append(Regression_out[1])
            inter.append(Regression_out[2])
            cl_accuracy.append(Regression_out[3])
            ncl_accuracy.append(Regression_out[4])
            lr = Regression_out[5]
            best_epochs.append(Regression_out[6])

        degree_of_polynom = '_'.join(str(d) for d in polynomial_degree)
        SAVING_PATH = f"{abs_path}/ResultsPolyRegression/" + dataset + f"/{num_encode_layers}el/{kshots}kshots/{degree_of_polynom}"
        with h5py.File(SAVING_PATH + FILENAME,'w') as f:
            f.create_dataset('regularization', data=np.arange(31))
            f.create_dataset('score', data=np.array(score))
            f.create_dataset('coefficients', data = np.array(coefficients))
            f.create_dataset('inter', data=np.array(inter))
            f.create_dataset('classical_accuracy', data=np.array(cl_accuracy))
            f.create_dataset('nonclassical_accuracy', data=np.array(ncl_accuracy))
            f.create_dataset('best_epoch', data=np.array(best_epochs))
        print(f'Saved results in {FILENAME}.')
    return None

if __name__=='__main__':
    # initialize dataset
    dataset = sys.argv[1]
    print('the dataset is: {}'.format(dataset))
    FILENAME = f"/CoefficientsAndAccuracyPolyRegression_M{1*1000}.hdf5"
    HierarchyOfRegression('.',FILENAME,dataset,[[1,0],[0,1], [1,1], [1,2], [2,1], [2,2]], 
                        decoder_architecture='14_2_1', num_encode_layers=1, kshots=1)
    HierarchyOfRegression('.',FILENAME,dataset[[1,0],[0,1], [1,1], [1,2], [2,1], [2,2]], 
                        decoder_architecture='14_2_1', num_encode_layers=1, kshots=100)