import jax
import jax.numpy as jnp
import numpy as np
from jax import random, vmap
from flax import linen as nn
from flax.typing import Initializer
from flax.linen.module import Module
from sympy import symbols
from sympy2jax import sympy2jax
from jax.tree_util import tree_structure
from polynom_skelleton import *

key = jax.random.PRNGKey(seed=0)

class QuantumAttentionNet(nn.Module):
    M : int # number of snapshot
    num_modes : int # number of modes
    num_layers : int # order of highest correlation
    decoder_fn: callable # decoder function used in the decoder
    kernel_init : Initializer = jax.nn.initializers.glorot_normal()

    def encode(self, k, x, s, layer):
        """ 
        input:
            k = matrix of the current step, has dim (d_h, d_x, d_x)
            x = original input
            s = sum of the previous layer
        output:
            y = output of the current layer
            s = sum of the current layer
        """
        if layer == 0:
            s = jnp.einsum('ne,...na,...ea -> ...na', k,x,x) # has shape (d_h, d_x, M)
            y = jnp.mean(s, axis=-1) # has shape (d_h, d_x)
        else:
            s = jnp.einsum('ne,...na,...ea -> ...na', k,x,s)
            y = jnp.mean(s, axis=-1)
        return y, s
    
    def Encoder(self, params,x):
        mean = jnp.mean(x, axis=-1)
        y,s = self.encode(k=params['k0'], x=x, s=x, layer=0)
        correlations = [mean,y]
        for l in range(1,self.num_layers):
            y, s = self.encode(params[f'k{l}'], x, s, layer=l)
            correlations.append(y)
        return jnp.array(correlations)
    
    def AlgebraicDecoder(self, params, corrs):
        ''' 
        input:
            params = dictionary with parameters from which the Decoder only uses theta
            corrs  = output of the decoder, having a shape (num_encoding_layers, 1, num_modes)
        ouput:
            y      = prediction of the model (0=classical, 1=non-classical)
        '''
        theta = params['theta']
        i = params['intercept']
        a = params['amplify']
        corrs = corrs.reshape((-1,self.num_modes*(self.num_layers+1)))# first dim has to be handled flexible
        h = self.decoder_fn(corrs, theta) + i
        return 1-nn.sigmoid(a*h)
    
    def init_params(self, key, modes, num_encode_layers):
        # encoder init
        w = jnp.array(jnp.eye(modes)) + 1e-3*jax.random.uniform(key,(modes,modes))
        tmp = {f'k{i}': jnp.array(w) for i in range(self.num_layers)}
        key, key1 = jax.random.split(key)
        # decoder init
        init_x = symbols(f'x:{modes*(num_encode_layers+1)}', positive=True)
        _, _, (_, initial_thetas) = jaxdecoder(init_x,modes,num_encode_layers)
        
        intercept = jax.random.uniform(key1)
        amplification = jax.random.uniform(key, minval=0.0, maxval=0.5)
        tmp.update({'theta': initial_thetas, 'intercept': intercept, 'amplify': amplification})
        return tmp
    
    def __call__(self, params, x):
        params = params['params']
        corrs = self.Encoder(params,x)
        y = self.AlgebraicDecoder(params, corrs)
        return y