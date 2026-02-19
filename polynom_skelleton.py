import jax, functools, itertools, multiprocessing, warnings, secrets
import numpy as np
import jax.numpy as jnp
from jax import random
from absl import flags, app
from flax import linen as nn
from flax.typing import Initializer
from flax.linen.module import Module
from sympy import symbols
from sympy2jax import sympy2jax
import hyperparams
from jax.tree_util import tree_structure

FLAGS = flags.FLAGS
jax.config.update("jax_traceback_filtering", "off")
app.parse_flags_with_usage(['.'])
key = jax.random.PRNGKey(seed=0)
rng1 = np.random.default_rng(seed=42)

def jax_trunc_poly(x:jnp.array, l:int, modes:int, num_encode_layers:int) -> jnp.array:
        ''' input:
                x = flattened list of correlation entries 
                        (encodinglayers + 1)*num_modes for diagonal encoder
                        (encodinglayers*num_modes + 1)*num_modes for non-diagonal encoder
                l = current layer
            output:
                polys = array of shape (num_modes,encode_layers + 1) containing all possible monomials of x^(l)
        '''
        y = x[:][l*(modes):(l+1)*(modes)]
        trunc_dim = int((num_encode_layers+1)/(l+1))
        polys = [[y[m]**i for i in range(1,trunc_dim+1)] for m in range(modes)]   
        return np.pad(np.array(polys), ((0,0),(0, num_encode_layers+1-trunc_dim)), mode='constant', constant_values=-5 )

def jaxdecoder(x:jnp.array, modes:int, num_encode_layers:int) -> (list, list, jnp.array):
    ''' input:
            x = correlations coming from the encoder, have (always) shape 
                    (num_encoding layers, 1, num_modes)
        output:
            sequence of shape (encode_layers+1, num_modes, encode_layers+1)
    '''    
    def superind2row(ind, modes,num_encode_layers):
        i = ind //(modes*(num_encode_layers+1)) # computes the encode layer
        j = (ind - i*(modes*(num_encode_layers+1)))//(num_encode_layers+1) # computes the mode
        return i,j
    def superind2corr_order(ind,modes, num_encode_layers):
        i = ind //(modes*(num_encode_layers+1))
        k = (ind - i*modes*(num_encode_layers+1))%(num_encode_layers+1)
        if i == 0:
            return k+1
        else:
            return (i+1)**(k+1) 
    if num_encode_layers < 1 or modes <1:
        raise ValueError("number of modes and number of encoding layers has to be >=1")
    polys = []   
    p = []
    if FLAGS.diagonal == True:
        for l in range(num_encode_layers+1):
            polys.append(jax_trunc_poly(x,l,modes,num_encode_layers))
        polys_flattened = np.concatenate(polys).flatten()
        #print(f'{polys_flattened=}')
        for l in range(1,num_encode_layers+2):
            combis = itertools.combinations(range(modes*((num_encode_layers+1)**2)), l)
            for c in combis:
                if l == 1:
                    if polys_flattened[c] > 0:
                        p.append(polys_flattened[c])
                else:
                    if sum([superind2corr_order(m,modes,num_encode_layers) for m in c]) > num_encode_layers+1:
                        continue
                    diff = [] 
                    diff_j = [] 
                    for b,d in itertools.combinations(c,2):
                        i1, j1 = superind2row(b,modes,num_encode_layers)
                        i2, j2 = superind2row(d,modes,num_encode_layers)
                        diff_j.append(abs(j1-j2))
                        diff.append(abs(i1-i2)+ abs(j1-j2))
                    if min(diff) == 0 or max(diff_j) > 0: # make sure, all elements correspond to the same mode
                        continue
                    elif sum([superind2corr_order(m,modes,num_encode_layers) for m in c]) <= num_encode_layers+1:
                        tmp = 1
                        for ci in c:
                            tmp *= polys_flattened[ci]
                        if tmp > 0:
                            p.append(tmp)
        #thetas = np.array([0.5+0.001*i for i in range(len(p))])
        thetas = rng1.uniform(low=-1.0, high=1.0,size=len(p))
        tmp = np.array(thetas*p)
        print(f'the monomials read {p}')
        expression = np.sum(thetas*p, axis=-1)
        print(f'the expression reads {expression}')
        perm = []
        for t in tmp:
            perm.append(np.where(np.array(expression.args)==t)[0][0])
        return p, perm, sympy2jax(expression, x)
    else:
        for l in range(num_encode_layers+1):
            polys.append(jax_trunc_poly(x,l,modes,num_encode_layers))
        polys_flattened = np.concatenate(polys).flatten()
        for l in range(1,num_encode_layers+2):
            combis = itertools.combinations(range(modes*((num_encode_layers+1)**2)), l)
            for c in combis:
                if l == 1:
                    if polys_flattened[c] > 0:
                        p.append(polys_flattened[c])
                else:
                    if sum([superind2corr_order(m,modes,num_encode_layers) for m in c]) > num_encode_layers+1:
                        continue
                    diff = [] 
                    for b,d in itertools.combinations(c,2):
                        i1, j1 = superind2row(b,modes,num_encode_layers)
                        i2, j2 = superind2row(d,modes,num_encode_layers)
                        diff.append(abs(i1-i2)+ abs(j1-j2))
                    if min(diff) == 0: # make sure, to skip diagonal terms
                        continue
                    elif sum([superind2corr_order(m,modes,num_encode_layers) for m in c]) <= num_encode_layers+1:
                        tmp = 1
                        for ci in c:
                            tmp *= polys_flattened[ci]
                        if tmp > 0:
                            p.append(tmp)
        thetas = np.array([0.5+0.001*i for i in range(len(p))])
        thetas = rng1.uniform(low=-1.0, high=1.0,size=len(p))
        
        tmp = np.array(thetas*p)
        expression = np.sum(thetas*p, axis=-1)
        perm = []
        print(f'the outputed monomials read {p}\n')
        print(f'the expression reads {expression}')
        # function that bookkeeps the order of the thetas for any initialization
        for t in tmp:
            perm.append(np.where(np.array(expression.args)==t)[0][0])
        return p, perm, sympy2jax(expression, x)