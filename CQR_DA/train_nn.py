from fire import Fire
import os
import pickle
import numpy as np
import tensorflow as tf
import keras
from keras.optimizers import Adam
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from networks import fully_convolutional_two_heads
from parameters import *  # uses nx, etc.

def limit_mem():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

def load(fn):
    with open(fn, 'rb') as f:
        return pickle.load(f, encoding='latin1')

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def make_256(x, newsize):
    a = np.empty((x.shape[0], newsize, x.shape[2]), 'float32')
    dsize = int((newsize - nx) / 2)
    a[:, dsize:-dsize] = x
    a[:, :dsize] = x[:, -dsize:]
    a[:, -dsize:] = x[:, :dsize]
    return a

def scale(x, m, s): return (x - m) / s
def unscale(x, m, s): return x * s + m

def save(mean, std, name):
    os.makedirs(name, exist_ok=True)
    np.save(name + 'mean.npy', mean)
    np.save(name + 'std.npy', std)

def get_data(path, exps=50, time_range=[0], ens=10, split=0.5, newsize=256):
    n = len(exps) * ens * len(time_range)   # one sample per member
    X = np.zeros((n, nx, 4), dtype=np.float32)
    Y = np.zeros((n, nx, 3), dtype=np.float32)

    for iexp, exp in enumerate(exps):
        exp_str = str(exp)
        qp_path  = os.path.join(path, exp_str, 'QPEns_data.pkl')
        tr_path  = os.path.join(path, exp_str, 'train_data.pkl')
        obs_path = os.path.join(path, exp_str, 'obs_pos.pkl')

        for pth in (qp_path, tr_path, obs_path):
            if not os.path.exists(pth):
                raise FileNotFoundError(f"Missing file: {pth}")

        Y_temp = load(qp_path)['analysis']   # dict: t -> (3*nx, k)
        X_an   = load(tr_path)               # dict: t -> (3*nx, k)
        X_obs  = load(obs_path)              # dict: t -> (3*nx,)

        for ti, t in enumerate(time_range):
            row0 = iexp * ens * len(time_range) + ti * ens
            row1 = row0 + ens
            Y[row0:row1, :, :] = np.asarray(Y_temp[t]).reshape(3, nx, -1).T
            X[row0:row1, :, :3] = np.asarray(X_an[t]).reshape(3, nx, -1).T
            X[row0:row1, :, 3] = np.tile(X_obs[t][2*nx:], (ens, 1))

    mean_temp = np.average(np.average(X[..., 0:3], axis=0), axis=0)
    mean = [mean_temp[0], mean_temp[1], mean_temp[2]]
    mean[2] = 0.0  # force r mean to 0
    var_temp = np.average(np.var(X[..., 0:3], axis=0), axis=0)
    std = [var_temp[0] ** 0.5, var_temp[1] ** 0.5, var_temp[2] ** 0.5]
    for i in range(3):
        X[..., i] = (X[..., i] - mean[i]) / std[i]
        Y[..., i] = (Y[..., i] - mean[i]) / std[i]

    X = make_256(X, newsize)

    split_idx = int(X.shape[0] * split)
    return X[:split_idx], X[split_idx:], Y[:split_idx], Y[split_idx:], mean, std

def _parse_list_like(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    s = str(x).strip()
    if s.startswith('[') or s.startswith('('):
        s = s[1:-1]
    if not s:
        return []
    return [int(v) for v in s.split(',')]

def _parse_taus(taus):
    if isinstance(taus, (list, tuple, np.ndarray)):
        return [float(t) for t in taus]
    s = str(taus).strip()
    if s.startswith('[') or s.startswith('('):
        s = s[1:-1]
    return [float(t) for t in s.split(',')]

def main(datadir,
         keras_save_fn,
         name,
         exps,
         nn_args,
         split=0.5,
         time_range=[20, 499],
         ens=10,
         lr=1e-3,
         epochs=10,
         bs=32,
         taus="0.05,0.95"):
    limit_mem()

    if isinstance(exps, str):
        exps_list = _parse_list_like(exps)
    else:
        exps_list = list(exps)
    if isinstance(time_range, str):
        tr = _parse_list_like(time_range)
    else:
        tr = list(time_range)
    taus_list = _parse_taus(taus)

    if isinstance(nn_args, str):
        import json
        try:
            nn_args_dict = json.loads(nn_args.replace("'", '"'))
        except Exception:
            nn_args_dict = eval(nn_args, {"__builtins__": {}}, {})
    else:
        nn_args_dict = dict(nn_args)

    newsize = nx + sum(nn_args_dict['kernels']) - len(nn_args_dict['kernels'])

    X_train, X_valid, Y_train, Y_valid, mean, std = get_data(
        datadir, exps_list, list(range(tr[0], tr[1] + 1)), ens, split, newsize
    )
    print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"X_valid: {X_valid.shape}, Y_valid: {Y_valid.shape}")
    save(mean, std, f'{keras_save_fn}{name}/')
    model = fully_convolutional_two_heads(
        filters=nn_args_dict['filters'],
        kernels=nn_args_dict['kernels'],
        positive_r=nn_args_dict.get('positive_r', True),
        inn=int(X_train.shape[-1])
    )

    tau_lo = taus_list[0]   
    tau_hi = taus_list[1]  

    def loss_lower(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(tau_lo * e, (tau_lo - 1.0) * e))

    def loss_upper(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(tau_hi * e, (tau_hi - 1.0) * e))

    model.compile(
        optimizer=Adam(lr),
        loss={'lower': loss_lower, 'upper': loss_upper}
    )

    os.makedirs(f'{keras_save_fn}{name}', exist_ok=True)
    filepath = f'{keras_save_fn}{name}/weights.best.keras'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=False, save_weights_only=False, mode='auto')

    histo = model.fit(
        X_train,
        {'lower': Y_train, 'upper': Y_train},
        batch_size=bs, epochs=epochs,
        validation_data=(X_valid, {'lower': Y_valid, 'upper': Y_valid}),
        shuffle=True, callbacks=[checkpoint]
    )
    save_obj(histo.history, f'{keras_save_fn}{name}/hist')
    model.save(f'{keras_save_fn}{name}/model.keras')


if __name__ == '__main__':
    Fire(main)