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

from main_DenseDecoder import *
from analyzing_the_model import counting_thetas
from model_DenseDecoder import *

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
    opt, state, params, QuAttnNet = initialize_model()
    layer_dims = list(map(int, FLAGS.layer_dims))
    for idx, l in enumerate(layer_dims[:-1]):
        if idx==0:
            jnp.array(params['weights_decoder'][0]).shape == (l,FLAGS.num_encode_layers+1)
        else:
            assert jnp.array(params['weights_decoder'][idx]).shape == (l, layer_dims[idx-1])
            assert jnp.array(params['bias_decoder'][idx]).shape == (l,)

@pytest.mark.parametrize("regu, suppress_k",[
    (0.,0.),
    (0.1,0),
    (0.,0.1),
    (0.1,0.1)
    ])

def test_main_training_and_saving(regu, suppress_k):
    if FLAGS.modes ==1:
        current_dataset='ds15_8Bin'
    elif FLAGS.modes==6:
        current_dataset='ds16_6modes'
    train_ds, test_ds= load_datasets(current_dataset)
    QuAttnNet = PreSymbolicRegressionNet(M=FLAGS.shots, 
                                    num_modes=FLAGS.modes, 
                                    num_layers=FLAGS.num_encode_layers,
                                   )
    params_model = QuAttnNet.init_params([14,4,1],rng,FLAGS.modes,FLAGS.num_encode_layers)['params']
    opt, state, params, QuAttnNet = initialize_model()
    new_params, state, cl, ncl, loss_train, loss_test=main_training(rng,opt,state,params,QuAttnNet, 20, train_ds, test_ds,regu, suppress_k)
    #print(jax.tree_util.tree_map(lambda x,y: x.shape==y.shape, params, new_params))
    _=save_encoder_outputs_and_predictions(QuAttnNet, params_model, train_ds, current_dataset, regu, suppress_k,"/Testfile.hdf5")
    filepath="SymbolicRegressionWithDenseDecoder/EncoderOutputsAndModelsPrediction/" \
        + current_dataset + f"/{FLAGS.num_encode_layers}el/{int(FLAGS.shots/1000)}kshots/"    
    
    assert os.path.exists(filepath+"Testfile.hdf5")
    os.remove(filepath+"Testfile.hdf5")