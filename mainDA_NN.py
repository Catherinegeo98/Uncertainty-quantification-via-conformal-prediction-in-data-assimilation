from parameters import *
from msw_model import sweq, initialize, noise_u
import assimilation as assim
import sys
import numpy as np
import random
import os.path
import os
import pickle
from tqdm import tqdm

CP_PATHS = [
    r"/home/mga5500/Schreibtisch/PYTHON/wavestoweather_CP_NN/cp_global_ensemble_200_full.pkl"
]

def load_cp_scores(paths):
    scores = {}
    for p in paths:
        try:
            with open(p, "rb") as f:
                scores = pickle.load(f)
            print(f"[CP] Loaded conformal scores: {p}")
            break
        except Exception as e:
            print(f"[CP] Could not load {p}: {e}")
    if not scores:
        print("[CP] WARNING: No CP scores loaded. All CP=0.")
    return scores

conformal_scores_all = load_cp_scores(CP_PATHS)


def cp_for_cycle(scores, ti):
    entry = scores.get(ti)
    if entry is None:
        entry = scores.get(str(ti))
    if entry is None:
        return {'u': 0.0, 'h': 0.0, 'r': 0.0}
    return {
        'u': float(entry.get('u', 0.0)),
        'h': float(entry.get('h', 0.0)),
        'r': float(entry.get('r', 0.0)),
    }

def add_cp_random_interval_scalar(E, cp, seed=None):
    rng = np.random.default_rng(seed)
    nrows, k = E.shape
    nx_local = nrows // 3

    cp_vec = np.empty(nrows, dtype=float)
    cp_vec[0:nx_local] = float(cp['u'])
    cp_vec[nx_local:2*nx_local] = float(cp['h']) 
    cp_vec[2*nx_local:3*nx_local] = float(cp['r'])

    cp_mat = cp_vec[:, None]
    N2 = rng.normal(0.0, 1.0, size=E.shape)
    perturb = (cp_mat * N2)/2

    return E + perturb


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


data = {}
data['truth'] = {}
data['obs_pos'] = {}

for method in methods:
    data[method] = {}
    data[method]['analysis'] = {}
    data[method]['first guess'] = {}

    data[method]['analysis_cp'] = {}

    if method == 'NN':
        from nn_assim import *


if training:
    data['train'] = {}


rd = 0  

for seed in range(1, 201):

    data = {}
    data['truth'] = {}
    data['obs_pos'] = {}

    for method in methods:
        data[method] = {}
        data[method]['analysis'] = {}
        data[method]['first guess'] = {}

        data[method]['analysis_cp'] = {}

        if method == 'NN':
            from nn_assim import *

    if training:
        data['train'] = {}

    r = [random.Random(10000 + seed * 100000 + s * 1000 + rd) for s in range(99)]

    datadir_seed = datadir + f'nsub{nsub}/nu{nu}/{k}/Data_DA_SCP_NN/{seed}/'
    if not os.path.exists(datadir_seed):
        os.makedirs(datadir_seed)

    obs_position = np.ones((mf * nx))
    ens = {}

    unoise_t = noise_u(r[0], 1)
    truth = initialize(unoise_t, 1)

    unoise = noise_u(r[1], k)
    ens_train = initialize(unoise, k)

    for method in methods:
        ens[method] = np.copy(ens_train)

    for ti in tqdm(range(cyc - 1)):

        unoise_t = noise_u(r[0], 1)
        unoise   = noise_u(r[1], k)

        truth = sweq(unoise_t, 1, truth)
        data['truth'][ti] = truth

        for method in methods:
            ens[method] = sweq(unoise, k, ens[method])
            data[method]['first guess'][ti] = ens[method].copy()

        obs, obs_original = assim.genobs(r[2], r[3], np.copy(truth))
        if radarint == 1:
            obs_position = assim.radar(np.copy(truth), r[4])
        data['obs_pos'][ti] = obs_position

        for method in methods:

            if method == 'QPEns' and training:
                ens[method], data['train'][ti] = assim.assimilation(
                    ens[method], obs_position, obs, method
                )
            else:
                ens[method] = assim.assimilation(
                    ens[method], obs_position, obs, method
                )

            if method == 'NN' and 'EnKF' in methods and ti >= 21:
                ens['NN'] = ens['EnKF'].copy()

            if method == 'NN' and ti >= 20:
                print('Start NN')
                ens['NN'] = nn_assim(ens['NN'], obs_position)
                print('Stop NN')

            data[method]['analysis'][ti] = ens[method].copy()

            if method == 'NN' and ti >= 20 and 'EnKF' in methods:
                nn_bg = np.copy(data['NN']['analysis'][ti]) 
                cp_t  = cp_for_cycle(conformal_scores_all, ti)
                nn_bg_cp = add_cp_random_interval_scalar(
                    nn_bg, cp_t, seed=ti )

                data['EnKF']['analysis_cp'][ti] = nn_bg_cp.copy()
                ens['EnKF'] = nn_bg_cp.copy()

    save_obj(data['truth'], f'{datadir_seed}truth_data')
    save_obj(data['obs_pos'], f'{datadir_seed}obs_pos')

    for method in methods:
        save_obj(data[method], f'{datadir_seed}{method}_data')

    if training:
        save_obj(data['train'], f'{datadir_seed}train_data')
