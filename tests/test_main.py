import jax, os, sys
import jax.numpy as jnp
import numpy as np
from jax import random
from flax import linen as nn
import pytest

SCRIPTPATH = os.path.abspath(__file__)
REPODIR = os.path.split(os.path.split(SCRIPTPATH)[0])[0]
sys.path.append(REPODIR)

rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)

from main import *
from analyzing_the_model import counting_thetas

def test_load_datasets():
    if FLAGS.modes==1:
        with pytest.raises(Exception,match=f"Please set FLAGS.modes to 6"):
            _,_=load_datasets('ds16_6modes')
    elif FLAGS.modes==6:
        with pytest.raises(Exception,match=f"Please set FLAGS.modes to 1"):
            _,_=load_datasets('ds15_8Bin')
    if FLAGS.batch_size != 68:
        with pytest.raises(Exception,match=f"To implement Batch Gradient Descent, set FLAGS.batch_size to 68"):
            _,_=load_datasets('ds12_fixed')

def test_initialize_model():
    opt, state, params, permu, QuAttnNet = initialize_model()
    if FLAGS.dense_decoder==False:
        assert len(jnp.array(params['theta']))+1==counting_thetas.number_thetas(1,FLAGS.num_encode_layers+1)

@pytest.mark.parametrize("regu, suppress_k",[
    (0.,0.),
    (0.1,0),
    (0.,0.1),
    (0.1,0.1)
    ])
def test_main_training(regu, suppress_k):
    if FLAGS.modes ==1:
        train_ds,test_ds=load_datasets('ds15_8Bin')
    elif FLAGS.modes==6:
        train_ds, test_ds= load_datasets('ds16_6modes')
    opt, state, params, permu, QuAttnNet = initialize_model()
    new_params, state, cl, ncl =main_training(rng,opt,state,params,QuAttnNet, 20, train_ds, test_ds,regu, suppress_k)
    #print(jax.tree_util.tree_map(lambda x,y: x.shape==y.shape, params, new_params))
    for key in params:
        assert jax.tree_util.tree_map(lambda x,y: x.shape==y.shape, params, new_params)[key]
