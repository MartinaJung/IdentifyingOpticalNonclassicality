import jax, os, sys
import jax.numpy as jnp
import numpy as np
from jax import random
from flax import linen as nn
import pytest

SCRIPTPATH = os.path.abspath(__file__)
REPODIR = os.path.split(os.path.split(SCRIPTPATH)[0])[0]
sys.path.append(REPODIR)

from analyzing_the_model import counting_thetas
from model_AlCla import *
from model_DenseDecoder import *
from polynom_skelleton import *
from sympy import symbols
from sympy2jax import sympy2jax

key = jax.random.PRNGKey(seed=0)

#################################################
#
# test Classifier with dense decoder
#
#################################################

@pytest.mark.parametrize("dx,L,layers",[
    (0,1,[14,4,1]),
    (1,1,[14,4,1]),
    (1,2,[14,2,1]),
    (1,3,[10,2,1]),
    (2,1,[10,4,1]),
    (2,2,[10,4,1]),
    (2,3,[14,4,1])
    ])

def test_params_init_dense_decoder(dx,L,layers):
    if min(dx,L)<1:
        with pytest.raises(ValueError,match="number of modes and number of encoding layers has to be >=1"):
            DenseDecoder = PreSymbolicRegressionNet(1000, dx,L)
            params = DenseDecoder.init_params(layers,key, dx,L)
    else:
        DenseDecoder = PreSymbolicRegressionNet(1000, dx,L)
        params = DenseDecoder.init_params(layers,key, dx,L)
        for i in range(L):
            assert jnp.array(params['params'][f'k{i}']).shape == (dx, dx)
        for idx, l in enumerate(layers[:-1]):
            if idx==0:
                jnp.array(params['params']['weights_decoder'][0]).shape == (l,L+1)
            else:
                assert jnp.array(params['params']['weights_decoder'][idx]).shape == (l, layers[idx-1])
                assert jnp.array(params['params']['bias_decoder'][idx]).shape == (l,)

@pytest.fixture
def dx_L_layers():
    return 1,2,[14,4,1]

@pytest.fixture
def dense_decoder(dx_L_layers):
    ''' Creates fresh instance of the QuantumAttentionNet '''
    dx=dx_L_layers[0]
    L=dx_L_layers[1]
    return PreSymbolicRegressionNet(M=1000, num_modes=dx, num_layers=L)


def test_layers_of_DenseDecoder(dense_decoder,dx_L_layers):
    dx=dx_L_layers[0]
    L=dx_L_layers[1]
    layers=dx_L_layers[2]
    QuAttnNet=dense_decoder
    params = QuAttnNet.init_params(layers,key,modes=dx, num_encode_layers=L)
    x = jax.random.uniform(key,(dx,1000))#([[i]*10 for i in range(6)])
    out_model = QuAttnNet(params=params, x=x)
    corrs = QuAttnNet.Encoder(params=params['params'],x=x)

    mean =  jnp.sum(x, axis=-1)/1000 # mean
    y,s = QuAttnNet.apply(variables=params, k=params['params']['k0'], x=x, s=x, method=QuAttnNet.encode, layer=0) #output of first layer
    y2,s2 = QuAttnNet.apply(variables=params, k=params['params']['k1'], x=x, s=s, method=QuAttnNet.encode, layer=1) #output of second layer
    assert (np.isclose(corrs, jnp.array([mean,y,y2]))).all()
    out_manual = QuAttnNet.apply(variables=params, params=params['params'],corrs=jnp.array([mean, y, y2]), method=QuAttnNet.DenseDecoder)
    out_manual = 1-jax.nn.sigmoid(out_manual)
    assert out_manual == out_model