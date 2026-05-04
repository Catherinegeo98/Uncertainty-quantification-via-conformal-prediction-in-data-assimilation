import numpy as np
import keras
import tensorflow as tf

from parameters import *
from networks import *  
from train_nn import limit_mem, make_256, scale, unscale

limit_mem() 

def clamp_multi_r(z):
    B = tf.shape(z)[0]
    L = tf.shape(z)[1]
    C = tf.shape(z)[2]
    Q = C // 3  

    z4 = tf.reshape(z, (B, L, Q, 3))  
    u = z4[..., 0:1]
    h = z4[..., 1:2]
    r = tf.nn.softplus(z4[..., 2:3])   
    z4p = tf.concat([u, h, r], axis=-1)
    return tf.reshape(z4p, (B, L, 3 * Q))

def load_stats(savedir):
    mean = np.load(savedir + 'mean.npy')
    std  = np.load(savedir + 'std.npy')
    return mean, std

mean, std = load_stats(SAVEDIR)
from networks import apply_pos_r   

nn = keras.models.load_model(
    SAVEDIR + 'model.keras',
    compile=False,
    safe_mode=False,
    custom_objects={
        'clamp_multi_r': clamp_multi_r,
        'apply_pos_r': apply_pos_r,    
    },
)

def nn_assim(model, obspos):
    m = scale(np.reshape(model, (3, nx, -1)).T, mean, std)
    k_here = m.shape[0]
    obs_flag = np.tile(obspos[2*nx:].reshape(-1, 1), (k_here, 1, 1))
    full = np.concatenate((m, obs_flag), axis=2)
    full = make_256(full, nn.input_shape[1])
    preds_lo, preds_hi = nn.predict(full, batch_size=k_here, verbose=0)
    # [k, L, 3] each

    lo  = unscale(preds_lo, mean, std)  
    hi  = unscale(preds_hi, mean, std)   
    mid = 0.5 * (lo + hi)

    analysis = np.reshape(mid.transpose(2, 1, 0), (3*nx, -1))
    lower    = np.reshape(lo.transpose(2, 1, 0),  (3*nx, -1))
    upper    = np.reshape(hi.transpose(2, 1, 0),  (3*nx, -1))

    return analysis, lower, upper

if __name__ == '__main__':
    pass
