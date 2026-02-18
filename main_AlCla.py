import numpy as np
import jax, h5py, sys
import jax.numpy as jnp
from absl import flags, app
from flax import linen as nn
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import hyperparams
FLAGS = flags.FLAGS
app.parse_flags_with_usage(['.'])

rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)

from train_AlCla import *

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = "serif"

######################################
#
# Load ds & initialize model
#
######################################

def load_datasets(current_ds):
    DATASET_PATH = "datasets/" + current_ds
    DATAFILE = f'/{current_ds}_train_{int(FLAGS.shots/1000)}kshots.npz'
    ds = get_h5py_ds(DATASET_PATH+DATAFILE)
    train_ds, test_ds = get_train_test_set(ds, rng, 0.8)
    if jnp.shape(jnp.array(train_ds['images']))[1]!= FLAGS.modes:
        num_modes_ds = jnp.shape(jnp.array(train_ds["images"]))[1]
        raise Exception(f'Please set FLAGS.modes to {num_modes_ds}')
    if len(train_ds["labels"]) != FLAGS.batch_size:
        raise Exception(f'To implement Batch Gradient Descent, set FLAGS.batch_size to {len(train_ds["labels"])}')
    return train_ds, test_ds

def initialize_model():
    init_x = symbols(f'x:{FLAGS.modes*(FLAGS.num_encode_layers+1)}', positive=True)
    symb_expression, permu, (decoder_fn, try_theta_init) = jaxdecoder(init_x,FLAGS.modes,FLAGS.num_encode_layers)
    print(f'Theta init is {try_theta_init} and the reshuffled theta is {np.array(try_theta_init)[np.array(permu)]}\n')
    print('The used permu is', np.array(permu))
    QuAttnNet = QuantumAttentionNet(M=FLAGS.shots, 
                                    num_modes=FLAGS.modes, 
                                        num_layers=FLAGS.num_encode_layers,
                                        decoder_fn=decoder_fn)
    
    opt, state, params = create_train_state(init_rng, QuAttnNet,
                            M = FLAGS.shots,
                            num_modes=FLAGS.modes,
                            num_layers=FLAGS.num_encode_layers,
                            learning_rate=FLAGS.learning_rate)
    return opt, state, params, permu, QuAttnNet


def save_train_test_ds_as_hdf5(current_ds):
    SAVENAME = 'saved_params/' + current_ds + '/train_test_' + current_ds + \
        f"_M{FLAGS.shots}.hdf5"
    try:
        with h5py.File(SAVENAME, "x") as f:
            f.create_dataset("train_ds/images", data = train_ds['images'])
            f.create_dataset("train_ds/labels", data = train_ds['labels'])
            f.create_dataset("test_ds/images", data = test_ds['images'])
            f.create_dataset("test_ds/labels", data = test_ds['labels'])
        print('Saved train and test dataset in:', SAVENAME)
    except:
        print('Train and test dataset already exist. Won\'t save them again.')
    return 0

######################################
#
# Save predictions
#
######################################

def save_predictions(params,permu, classical_accuracy, nonclassical_accuracy, current_ds, regularization, suppress_k, *RESULTS_FILENAME):
    doc = {"SHOTS": FLAGS.shots, "MINIBATCH_SIZE": FLAGS.batch_size,
        "MODES": FLAGS.modes, "LEARNING_RATE": FLAGS.learning_rate,
        "NUM_ENCODE_LAYERS": FLAGS.num_encode_layers,
        "REGULARIZATION": regularization, "SUPPRESSING_K": suppress_k}
    SAVING_PATH = f"saved_params/{current_ds}/{FLAGS.num_encode_layers}el/{int(FLAGS.shots/1000)}kshots/"
    if len(RESULTS_FILENAME)==0:
        RESULTS_FILENAME = f"/Results_{current_ds}_RGZN{regularization}_BatchGD_lr{FLAGS.learning_rate}"\
                + f"_{int(FLAGS.shots/1000)}kshots_BestTrainEpoch.hdf5"
    else:
        RESULTS_FILENAME=RESULTS_FILENAME[0]
    # SAVE OPTIMAL PARAMETERS AND PERFORMANCE
    with h5py.File(SAVING_PATH + RESULTS_FILENAME, "w") as f:
        for key in params:
            if key == 'theta':
                f.create_dataset(key, data=params[key][np.array(permu)]) # save reshuffled theta
            else:
                try:
                    f.create_dataset(key, data=params[key])
                except:
                    for idx, element in enumerate(params[key]):
                        f.create_dataset(key + "/layer" + str(idx), data=element)

        f.create_dataset("cl_accuracy", data=round(classical_accuracy,4))
        f.create_dataset("ncl_accuracy", data=round(nonclassical_accuracy,4))
        f.create_dataset("best_epoch", data=FLAGS.epochs)
        f.attrs["DIAGONAL_ENCODER"]=FLAGS.diagonal
        f.attrs.update(doc)
    print(f'Saved accuracy and best epoch in {RESULTS_FILENAME}')
    return 0

######################################
#
# Train model for FLAGS.epochs epochs
#
######################################

def main_training(rng,opt,state,params,QuAttnNet, epochs, train_ds, test_ds,regularization, suppress_k):
    accuracy_on_train_set = []
    accuracy_on_test_set  = []
    loss_train_set        = []
    loss_test_set         = []
    cl_accracy            = []
    ncl_accracy           = []
    lr_scale_history = []

    for epoch in range(1, epochs + 1):
        rng, input_rng = jax.random.split(rng)

        state, params, train_loss, train_accuracy = train_test_epoch(opt, state, QuAttnNet,params,train_ds, input_rng,
                                            'train', regularization, suppress_k)
        test_loss, test_accuracy = train_test_epoch(opt, state, QuAttnNet,params, test_ds, input_rng, 
                                            'test', regularization, suppress_k)

        if epoch % 5 == 0:
            accuracy_on_train_set.append(train_accuracy)
            accuracy_on_test_set.append(test_accuracy)
            loss_train_set.append(train_loss)
            loss_test_set.append(test_loss)
            lr_scale = optax.tree.get(state, "scale")
            lr_scale_history.append(lr_scale)

            class_pred, non_class_pred = get_prediction_distribution(QuAttnNet,params, train_ds, regularization, suppress_k)
            classical_acc = jnp.sum(class_pred < 0.500)/len(class_pred)
            nonclassical_acc = jnp.sum(non_class_pred > 0.500)/len(non_class_pred)
            cl_accracy.append(classical_acc)
            ncl_accracy.append(nonclassical_acc)
            if epoch % 5 == 0:
                print(f'epoch {epoch} train accuracy: {train_accuracy}, test accuracy: {test_accuracy}, lr scale: {optax.tree.get(state, 'scale')}')
    print(f'Finished with epochs!')
    return params, state, jnp.array(cl_accracy), jnp.array(ncl_accracy), jnp.array(loss_train_set), jnp.array(loss_test_set)

###########################################
#
# Fix current dataset
# Initialize regularization strength
# Initialize regularization for K params
#
###########################################

if __name__ == "__main__":
    Current_dataset= sys.argv[1]
    regularization = float(sys.argv[2])
    if len(sys.argv) < 4:
        suppress_k = 0.0
    else:
        suppress_k = float(sys.argv[3])
    print('the regularization is: {}'.format(regularization))
    print('the suppressing factor is: {}'.format(suppress_k))

    train_ds, test_ds = load_datasets(Current_dataset)
    save_train_test_ds_as_hdf5(Current_dataset)
    opt, state, params, permu, QuAttnNet = initialize_model()
    
    print(f'parameters have shapes:', jax.tree_util.tree_map(lambda x: x.shape, params))

    params, state, cl_accracy, ncl_accracy, loss_train, loss_test = main_training(
                                                        rng, 
                                                        opt, 
                                                        state, 
                                                        params, 
                                                        QuAttnNet,
                                                        FLAGS.epochs, 
                                                        train_ds, 
                                                        test_ds,
                                                        regularization,
                                                        suppress_k)
    best_epoch = FLAGS.epochs
    classical_accuracy = cl_accracy[int(best_epoch/5)-1]
    nonclassical_accuracy = ncl_accracy[int(best_epoch/5)-1]
    print(f'classical accuracy: {classical_accuracy}')
    print(f'nonclassical accuracy: {nonclassical_accuracy}\n')

    epochs = jnp.arange(0,900,5)
    plt.plot(epochs,loss_train, label='training dataset')
    plt.plot(epochs,loss_test, label='test dataset')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    savename=f'Loss_curve_{Current_dataset}_{FLAGS.num_encode_layers}el_{FLAGS.shots}kshots'
    #plt.savefig(savename +'.pdf')
    #plt.savefig(savename +'.svg')
    plt.show()
    #save_predictions(params,permu,cl_accracy, ncl_accracy, Current_dataset,regularization, suppress_k, *FILENAME)
    pass
