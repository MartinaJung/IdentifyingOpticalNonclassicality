import jax, functools
import jax.numpy as jnp
import numpy as np
from jax import random, vmap
from flax import linen as nn
from flax.typing import Initializer
from flax.linen.module import Module
from jax.tree_util import tree_structure

key = jax.random.PRNGKey(seed=0)


class PreSymbolicRegressionNet(nn.Module):
    M : int # number of snapshots
    num_modes : int # number of modes
    num_layers : int # order of highest correlation
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
        y,s = self.encode(k=params['k0'], x=x, s=x, layer=0) # first encoding layer
        correlations = [mean,y]
        for l in range(1,self.num_layers):
            y, s = self.encode(params[f'k{l}'], x, s, layer=l)
            correlations.append(y)
        return jnp.array(correlations)
    
    def DenseDecoder(self, params, corrs):
        ''' 
        input:
            params = dictionary with parameters from which the Decoder only uses weights_decoder and bias_decoder
            corrs  = output of the decoder, having a shape (num_encoding_layers, 1, num_modes)
        ouput:
            y      = prediction of the model
        '''
        w = params['weights_decoder']
        b = params['bias_decoder']
        x = corrs
        for i, (wi,bi) in enumerate(zip(w,b)):
            x = jnp.einsum('nj,j... -> n', wi, x) + bi
            if i < len(w)-1:
                x = nn.relu(x)
        return x
        
    def init_params(self, layers, key, modes, num_encode_layers):
        if min(modes,num_encode_layers)<1:
            raise ValueError("number of modes and number of encoding layers has to be >=1")
        # encoder init
        w = jnp.array(jnp.eye(modes)) + 1e-3*jax.random.uniform(key,(modes,modes))
        tmp = {f'k{i}': jnp.array(w) for i in range(self.num_layers)}
        # decoder init
        key, key1 = jax.random.split(key)
        weights_decoder, bias_decoder = [],[]
        for j,l in enumerate(layers):
            if j == 0:
                weights_decoder.append(jax.random.uniform(key1,(l, num_encode_layers+1)))
            else:
                weights_decoder.append(jax.random.uniform(key1,(l, layers[j-1])))
            bias_decoder.append(jax.random.uniform(key, (l)))
        tmp.update({'weights_decoder': weights_decoder, 'bias_decoder':bias_decoder})         
        return {'params': tmp}
    
    def __call__(self, params, x):
        params = params['params']
        corrs = self.Encoder(params,x)
        x = self.DenseDecoder(params,corrs)
        return 1-nn.sigmoid(x)