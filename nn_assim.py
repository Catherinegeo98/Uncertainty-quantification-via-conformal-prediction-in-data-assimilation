import keras
import numpy as np
from parameters import *
from networks import *
from train_nn import limit_mem, make_256, scale, unscale
keras.losses.combined_loss = combined_bias_loss(1, 1.0)
limit_mem()  # Comment out if it throws error on CPU

def load(name):
    return np.load(name+'mean.npy'),np.load(name+'std.npy')

mean, std = load(SAVEDIR)
nn = keras.models.load_model(f'{SAVEDIR}model.keras',
    custom_objects={'combined_loss': combined_bias_loss(1, 1.0)})

def nn_assim(model,obspos):
    m = scale(np.reshape(model, (3, nx, -1)).T, mean, std) 
    full = make_256(np.concatenate((m,np.tile(obspos[2*nx:].reshape(-1,1),(k,1,1))),axis=2),nn.input_shape[1])    
    preds = unscale(nn.predict(full, full.shape[0]), mean, std)
    return np.reshape(preds.T,(3*nx,-1))


if __name__ == '__main__':
    nn_assim(None, None)
