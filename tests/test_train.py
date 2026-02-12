import jax, os, sys
import jax.numpy as jnp
import numpy as np
from jax import random
from flax import linen as nn
import pytest

SCRIPTPATH = os.path.abspath(__file__)
REPODIR = os.path.split(os.path.split(SCRIPTPATH)[0])[0]
sys.path.append(REPODIR)

from train import *
if FLAGS.dense_decoder:
    from model_DenseDecoder import *
else:
    from model import *

rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)

# test update
@pytest.mark.parametrize("ds",[
    "ds15_8Bin",
    "ds12_fixed",
    "ds14_PNR",
    "ds16_6modes"
    ])

def test_h5py_ds(ds):
    DATAFILE=f'{ds}_train_1kshots.npz'
    ds5 = get_h5py_ds(f"./datasets/{ds}/"+DATAFILE)
    assert jnp.array(ds5['images']).shape[-1]==1000
    assert jnp.array(ds5['labels']).shape[0]==jnp.array(ds5['images']).shape[0]
    # test get_train_test_ds
    train_ds, test_ds=get_train_test_set(ds5, rng, 0.8)
    assert jnp.array(train_ds['images']).shape[-1]==1000
    assert jnp.array(train_ds['labels']).shape[0]==jnp.array(train_ds['images']).shape[0]

def test_create_train_state():
    if FLAGS.dense_decoder:
        QuAttnNet = PreSymbolicRegressionNet(M=FLAGS.shots, 
                                    num_modes=FLAGS.modes, 
                                    num_layers=FLAGS.num_encode_layers,
                                    )
    else:
        init_x = symbols(f'x:{FLAGS.modes*(FLAGS.num_encode_layers+1)}', positive=True)
        symb_expression, permu, (decoder_fn, try_theta_init) = jaxdecoder(init_x,FLAGS.modes,FLAGS.num_encode_layers)
        QuAttnNet = QuantumAttentionNet(M=FLAGS.shots, 
                                    num_modes=FLAGS.modes, 
                                    num_layers=FLAGS.num_encode_layers,
                                    decoder_fn=decoder_fn)
    
    opt, state, params = create_train_state(init_rng, QuAttnNet,
                           M = FLAGS.shots,
                           num_modes=FLAGS.modes,
                           num_layers=FLAGS.num_encode_layers,
                           learning_rate=FLAGS.learning_rate)
    if FLAGS.dense_decoder:
        layer_dims = list(map(int, FLAGS.layer_dims))
        params_model = QuAttnNet.init_params(layer_dims,init_rng,FLAGS.modes,FLAGS.num_encode_layers)['params']
    else:
        params_model = QuAttnNet.init_params(init_rng,FLAGS.modes,FLAGS.num_encode_layers)
    for key in params:
        if key not in ['weights_decoder', 'bias_decoder']:
            assert (jnp.array(params[key]).shape == jnp.array(params_model[key]).shape)
            if key != 'theta':
                assert (jnp.isclose(jnp.array(params[key]),jnp.array(params_model[key]))).all()
        else:
            for idx, k in enumerate(params[key]):
                assert (jnp.array(k).shape == jnp.array(params_model[key][idx]).shape)
                assert (jnp.isclose(jnp.array(k),params_model[key][idx])).all()

###############################################
#
# test updates
# 
###############################################

@pytest.fixture
def optimizer_state():
    if FLAGS.dense_decoder:
        QuAttnNet = PreSymbolicRegressionNet(M=FLAGS.shots, 
                                    num_modes=FLAGS.modes, 
                                    num_layers=FLAGS.num_encode_layers,
                                   )
    else:
        init_x = symbols(f'x:{FLAGS.modes*(FLAGS.num_encode_layers+1)}', positive=True)
        symb_expression, permu, (decoder_fn, try_theta_init) = jaxdecoder(init_x,FLAGS.modes,FLAGS.num_encode_layers)
        QuAttnNet = QuantumAttentionNet(M=FLAGS.shots, 
                                    num_modes=FLAGS.modes, 
                                    num_layers=FLAGS.num_encode_layers,
                                    decoder_fn=decoder_fn)
    opt, state, params = create_train_state(init_rng, QuAttnNet,
                           M = FLAGS.shots,
                           num_modes=FLAGS.modes,
                           num_layers=FLAGS.num_encode_layers,
                           learning_rate=FLAGS.learning_rate)
    return opt, state, params, QuAttnNet

@pytest.fixture
def get_train_ds15():
    DATAFILE=f'ds15_8Bin_train_1kshots.npz'
    ds5 = get_h5py_ds(f"./datasets/ds15_8Bin/"+DATAFILE)
    train_ds, test_ds=get_train_test_set(ds5, rng, 0.8)
    return train_ds, test_ds

@pytest.mark.parametrize("regu, suppress_k",[
    (0.,0.),
    (0.1,0),
    (0.,0.1),
    (0.1,0.1)
    ])

def test_train_test_epoch(optimizer_state, get_train_ds15, regu, suppress_k):
    train_ds, test_ds = get_train_ds15
    opt, state, params, QuAttnNet = optimizer_state
    state, new_params, train_loss, train_accuracy = train_test_epoch(opt, state, QuAttnNet,
                                        params,train_ds, rng,
                                        'train', regu, suppress_k)
    test_loss, test_accuracy = train_test_epoch(opt, state, QuAttnNet,params, test_ds, rng, 
                                        'test', regu, suppress_k)
    for key in params:
        assert jax.tree_util.tree_map(lambda x,y: x.shape==y.shape, params, new_params)[key]
    
    # TODO: Test update (what is happening behind the curtain?)
    # cum_grads = {}
    # for idx, x in enumerate(train_ds['images']):
    #     grads, loss, accuracy, logits=apply_model(params, x, train_ds['labels'][idx], QuAttnNet, 0.,0.)
    #     if idx==0:
    #         for key in grads:
    #             cum_grads[key]=-grads[key]/39
    #     else:
    #         for key in grads:
    #             cum_grads[key]+=-grads[key]/39
    # params_manually_updated = optax.apply_updates(params,cum_grads)
    # print('manually updated params:', params_manually_updated['theta'])
    # for key in params:
    #     assert jnp.isclose(params_manually_updated[key], new_params[key]).all()
@pytest.mark.parametrize("regu, suppress_k",[
    (0.,0.),
    (0.1,0),
    (0.,0.1),
    (0.1,0.1)
    ])

def test_get_prediction_distribution(get_train_ds15,optimizer_state, regu, suppress_k):
    train_ds, test_ds = get_train_ds15
    opt, state, params, QuAttnNet = optimizer_state
    cl_preds, noncl_preds = get_prediction_distribution(QuAttnNet, params, train_ds, regu, suppress_k)
    assert len(cl_preds)+len(noncl_preds) == len(train_ds['labels'])
    if FLAGS.dense_decoder:
        encoder_out, preds, true_labels = get_predictions_and_encoder_outputs(QuAttnNet,params, train_ds, regu, suppress_k)
        assert len(preds) == len(cl_preds)+len(noncl_preds)
