import jax, optax, csv, functools
import jax.numpy as jnp
from absl import flags, app
import numpy as np
from jax import random
from functools import partial
from flax import linen as nn
import optax.tree
from optax import contrib
from jax.scipy.special import xlog1py, xlogy

import hyperparams
FLAGS = flags.FLAGS
app.parse_flags_with_usage(['.'])

from model_AlCla import *

# HYPERPARAMETERS
PATIENCE = 10 # Number of epochs with no improvement after which learning rate will be reduced
COOLDOWN = 0 # Number of epochs to wait before resuming normal operation after the learning rate reduction
FACTOR = 0.5 # Factor by which to reduce the learning rate
RTOL = 1e-4 # Relative tolerance for measuring the new optimum
ACCUMULATION_SIZE = 39 # Number of iterations to accumulate an average value


def get_h5py_ds(file:str) -> dict:
    data = np.load(file)
    cl_images = data['cl_images']
    ncl_images = data['ncl_images']
    images = np.concatenate((cl_images, ncl_images), axis=0)
    return {'images': images, 'labels': [0]*len(cl_images) + [1]*len(ncl_images)}

def create_train_state(key:jnp.array, QuAttnNet, num_modes:int, num_layers:int, learning_rate:float):
    params = QuAttnNet.init_params(key,num_modes,num_layers)
    opt = optax.chain(optax.adam(learning_rate),
            contrib.reduce_on_plateau(
                patience=PATIENCE,
                cooldown=COOLDOWN,
                factor=FACTOR,
                rtol=RTOL,
                accumulation_size=ACCUMULATION_SIZE,
            ),
        )
    opt_state = opt.init(params)
    return opt, opt_state, params

@partial(jax.jit, static_argnames=['QuAttnNet','regularization_strength', 'suppressing_k'])
def loss_fn(params:dict, 
        images:jnp.array, 
        label:jnp.array, 
        QuAttnNet,
        regularization_strength:float, 
        suppressing_k:float) -> (jnp.array, jnp.array):
        logits = QuAttnNet({"params": params}, images)
        def true_fc(l):
            return jnp.mean(optax.sigmoid_binary_cross_entropy(logits=jnp.log(l/(1-l)+1e-6), labels=label))
        def false_fc(l):
            return jnp.mean(-label*jnp.log(1.)- (1.-label)*jnp.log(1e-6))
        loss = jax.lax.cond(((1.-logits) >= 1e-6)[0], true_fc, false_fc, logits)
        regularization = jnp.sum((1.-label)*(logits**2))
        loss += regularization_strength*regularization
        for i in range(FLAGS.num_encode_layers):
            loss += suppressing_k*jnp.sum((params[f'k{i}'])**2)
        return loss, logits

@partial(jax.jit, static_argnames=['QuAttnNet','regularization_strength', 'suppressing_k'])
def apply_model(params:dict, 
        images:jnp.array, 
        label:jnp.array, 
        QuAttnNet, 
        regularization_strength:float, 
        suppressing_k:float) -> (jnp.array, jnp.array,jnp.array,jnp.array):
    " compute gradients, loss (binary cross entropy) and accuracy "    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(params, images, label, QuAttnNet, regularization_strength, suppressing_k)
    accuracy = (jnp.abs(logits-label*jnp.ones(logits.shape)) < 0.5)
    return grads, loss, accuracy, logits

@partial(jax.jit, static_argnames=['opt'])
def update_model(params:dict, opt, opt_state, grads:jnp.array, accuracy:jnp.array):
    updates, opt_state = opt.update(grads, opt_state, params, value=accuracy)
    params = optax.apply_updates(params, updates)
    return params, opt_state

def get_train_test_set(ds:dict, rng:jnp.array, ratio:float) -> (dict, dict):
    ds_size = len(ds['images'])
    perm = jax.random.permutation(rng, ds_size)
    
    len_train_ds = int(ratio*ds_size)  
    train_images = []
    test_images  = []
    train_labels = []
    test_labels  = []
    
    for idx in range(len_train_ds):
        train_images.append(ds['images'][perm[idx]])
        train_labels.append(ds['labels'][perm[idx]])
    for idx2 in range(len_train_ds, ds_size):
        test_images.append(ds['images'][perm[idx2]])
        test_labels.append(ds['labels'][perm[idx2]])    
    train_dataset = {'images': train_images, 'labels': train_labels}
    test_dataset = {'images': test_images, 'labels': test_labels}
    return train_dataset, test_dataset

def train_test_epoch(opt, state, 
                    QuAttnNet, 
                    params:dict, 
                    train_ds:dict, 
                    rng:jnp.array, 
                    phase:str, 
                    regularization_strength:float, 
                    suppressing_k:float):
    """ If phase =='test'
            output: loss, accuracy
        If phase =='train'
            output: state, loss, accuracy
    """
    train_ds_size = len(train_ds['images'])
    num_minibatches = train_ds_size//FLAGS.batch_size
    perm = jax.random.permutation(rng, train_ds_size)
    images_reshuffled = jnp.array(train_ds['images'])[perm]
    labels_reshuffled = jnp.array(train_ds['labels'])[perm]
    
    epoch_loss, epoch_accuracy, test_loss, test_accuracy = [],[],[],[]
    
    v_apply = jax.vmap(apply_model, in_axes=(None,0,0,None,None,None), out_axes=0)
    if phase == 'train':
        img_batches = jnp.reshape(images_reshuffled, (num_minibatches, FLAGS.batch_size, -1, FLAGS.shots))
        labels_batches = jnp.reshape(labels_reshuffled, (num_minibatches, FLAGS.batch_size, -1,))
        for img, labels_batch in zip(img_batches, labels_batches):
            grads, loss, accuracy, logits = v_apply(params, img, labels_batch, QuAttnNet,regularization_strength, suppressing_k)
            # average gradients
            average_over_batch = partial(jnp.mean, axis=0)
            grads_mean = jax.tree.map(average_over_batch, grads)          
            params, state = update_model(params, opt, state, grads_mean, jnp.mean(accuracy))

            # Clip k matrices with upper triangular
            if FLAGS.modes > 1:
                lower_k = -10.*jnp.triu(jnp.ones((FLAGS.modes, FLAGS.modes)))
                upper_k = 10.*jnp.triu(jnp.ones((FLAGS.modes, FLAGS.modes)))
            else:
                lower_k = -10.*jnp.ones((FLAGS.modes, FLAGS.modes)) # lower bound the K-values by -10
                upper_k = 10*jnp.ones((FLAGS.modes, FLAGS.modes))
            for l in range(FLAGS.num_encode_layers):
                params[f'k{l}'] = optax.projections.projection_box(params[f'k{l}'], lower=lower_k, upper=upper_k)

            # Clip amplitude and theta  
            low_amplify = 1.0
            up_amplify = 50.0
            params['amplify'] = optax.projections.projection_box(params['amplify'], lower=low_amplify, upper=up_amplify)
            lower_theta = jnp.array([-10.0]*len(params['theta']))
            upper_theta = jnp.array([ 10.0]*len(params['theta']))
            params['theta'] = optax.projections.projection_box(params['theta'], lower=lower_theta, upper=upper_theta)
            
            epoch_loss.append(jnp.mean(loss))
            epoch_accuracy.append(jnp.mean(accuracy))
        train_loss = np.mean(epoch_loss)
        train_accuracy = np.mean(epoch_accuracy)
        return state, params, train_loss, train_accuracy
    
    elif phase == 'test':
        for idx, img in enumerate(images_reshuffled):
            _, loss, accuracy, logits = apply_model(params, img, labels_reshuffled[idx], QuAttnNet,regularization_strength, suppressing_k)
            epoch_loss.append(loss)
            epoch_accuracy.append(accuracy)
        test_loss = np.mean(epoch_loss)
        test_accuracy = np.mean(epoch_accuracy)
        return test_loss, test_accuracy    

def get_prediction_distribution(QuAttnNet, 
                            params:dict, 
                            train_ds:dict, 
                            regularization_strength:float, 
                            suppressing_k:float) -> (jnp.array,jnp.array):
    classicals_prediction = []
    non_classicals_prediction = []
    labels = jnp.array(train_ds['labels'])
    _, logits = loss_fn(params, jnp.array(train_ds['images']), labels, QuAttnNet,regularization_strength, suppressing_k)
    classical_indices = jnp.where((labels<1))
    nonclassical_indices = jnp.where((labels>0))
    comp_classicals_prediction = logits[classical_indices]
    comp_nonclassical_prediction = logits[nonclassical_indices]

    for i in range(len(train_ds['images'])):
        label = jnp.array(train_ds['labels'])[i]
        if label == 0:
            _, logits = loss_fn(params, train_ds['images'][i], 0, QuAttnNet,regularization_strength, suppressing_k)
            classicals_prediction.append(jnp.mean(logits))
        else:
            _, logits = loss_fn(params, train_ds['images'][i], 1, QuAttnNet,regularization_strength, suppressing_k)
            non_classicals_prediction.append(jnp.mean(logits))
    return jnp.array(classicals_prediction), jnp.array(non_classicals_prediction)

def get_predictions_and_encoder_outputs(QuAttnNet,
                                        best_params:dict, 
                                        train_ds:dict, 
                                        regularization_strength:float, 
                                        suppressing_k:float) -> (jnp.array,jnp.array,jnp.array):
    encoder_outputs = []
    predictions = []
    true_labels = []
    QuAttnNet = PreSymbolicRegressionNet(M=FLAGS.shots, num_modes=FLAGS.modes, num_layers=FLAGS.num_encode_layers)
    for i in range(len(train_ds['images'])):
        true_labels.append(jnp.array(train_ds['labels'])[i])
        y = QuAttnNet.Encoder(best_params, train_ds['images'][i])
        encoder_outputs.append(y[:,0])
        logits_unnormalized = QuAttnNet.DenseDecoder(best_params,y)
        predictions.append(jnp.mean(logits_unnormalized))
    return jnp.array(encoder_outputs), jnp.array(predictions), jnp.array(true_labels)
