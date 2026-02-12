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
from model import *
from model_DenseDecoder import *
from polynom_skelleton import *
from sympy import symbols
from sympy2jax import sympy2jax

key = jax.random.PRNGKey(seed=0)

#################################################
#
# test polynomial skelleton
#
#################################################

@pytest.mark.parametrize("dx,L",[
    (0,3),
    (0,0),
    (1,1),
    (2,1),
    (2,2),
    (2,3),
    (1,1),
    (1,2),
    (1,3),
])
def test_polynomial_skelleton(dx,L):
    if dx<1 or L<1:
        with pytest.raises(ValueError,match="number of modes and number of encoding layers has to be >=1"):
            init_x = symbols(f'x:{0}', positive=True)
            symb_expression, permu, (decoder_fn, try_theta_init) = jaxdecoder(init_x,dx,L)
    else:
        init_x = symbols(f'x:{dx*(L+1)}', positive=True)
        symb_expression, permu, (decoder_fn, try_theta_init) = jaxdecoder(init_x,dx,L)
        assert len(permu) +1 == counting_thetas.number_thetas(dx,L+1)


#################################################
#
# test Algebraic Classifier
#
#################################################
    
@pytest.mark.parametrize("dx,L",[
    (0.,0.),
    (1,0),
    (1,1),
    (1,2),
    (1,3),
    (2,1),
    (2,2),
    (2,3)
])
def test_params_init(dx, L):
    if min(dx,L)<1:
        print('I am checking the case dx=0')
        with pytest.raises(ValueError,match="number of modes and number of encoding layers has to be >=1"):
            init_x = symbols(f'x:{dx*(L+1)}', positive=True)
            symb_expression, permu, (decoder_fn, try_theta_init) = jaxdecoder(init_x,dx,L)
            QuAttnNet = QuantumAttentionNet(1000, dx,L,decoder_fn)
    else:
        init_x = symbols(f'x:{dx*(L+1)}', positive=True)
        symb_expression, permu, (decoder_fn, try_theta_init) = jaxdecoder(init_x,dx,L)
        QuAttnNet = QuantumAttentionNet(1000, dx,L,decoder_fn)
        params = QuAttnNet.init_params(key, dx,L)
        for i in range(L):
            assert jnp.array(params[f'k{i}']).shape == (dx, dx)
        assert jnp.array(params['theta']).shape[0]+1 == counting_thetas.number_thetas(dx,L+1)
        assert len(jnp.array(params['amplify']).flatten()) == 1
        assert len(jnp.array(params['intercept']).flatten())==1

@pytest.fixture
def dx_L():
    return 6,2

@pytest.fixture
def quantum_attention_net(dx_L):
    ''' Creates fresh instance of the QuantumAttentionNet '''
    dx=dx_L[0]
    L=dx_L[1]
    init_x = symbols(f'x:{dx*(L+1)}', positive=True)
    symb_expression, permu, (decoder_fn, try_theta_init) = jaxdecoder(init_x,dx,L)
    return QuantumAttentionNet(M=1000, num_modes=dx, num_layers=L, decoder_fn=decoder_fn)

def test_layers_of_AlCla(quantum_attention_net, dx_L):
    dx=dx_L[0]
    L=dx_L[1]
    QuAttnNet=quantum_attention_net
    params = QuAttnNet.init_params(key,modes=dx, num_encode_layers=L)
    x = jax.random.uniform(key,(dx,1000))#([[i]*10 for i in range(6)])
    out_model = QuAttnNet(params={'params': params}, x= x) # correlations after one layer
    corrs = QuAttnNet.Encoder(params=params, x=x)
    
    mean =  jnp.sum(x, axis=-1)/1000 # mean
    y,s = QuAttnNet.apply(variables={'params': params}, k=params['k0'], x=x, s=x, method=QuAttnNet.encode, layer=0) #output of first layer
    y2,s2 = QuAttnNet.apply(variables={'params': params}, k=params['k1'], x=x, s=s, method=QuAttnNet.encode, layer=1) #output of second layer
    assert (jnp.isclose(corrs,jnp.array([mean, y,y2]))).all() # check encoder outputs
    out_manual = QuAttnNet.apply(variables={'params': params}, params=params,corrs=jnp.array([mean, y,y2]), method=QuAttnNet.AlgebraicDecoder)
    assert jnp.isclose(out_manual,out_model)
    assert FLAGS.diagonal==False

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
        for idx, l in enumerate(layers[:-2]):
            if idx==0:
                jnp.array(params['params']['weights_decoder'][0]).shape == (l,L+1)
            else:
                assert jnp.array(params['params']['weights_decoder'][idx]).shape == (layers[idx+1],l)
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
    if min(dx,L)<1:
        with pytest.raises(ValueError,match="number of modes and number of encoding layers has to be >=1"):
            params = QuAttnNet.init_params(layers,key,modes=dx, num_encode_layers=L)
            x = jax.random.uniform(key,(dx,1000))#([[i]*10 for i in range(6)])
            out_model = QuAttnNet(params=params, x=x)
            corrs = QuAttnNet.Encoder(params=params['params'],x=x)
    else:
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