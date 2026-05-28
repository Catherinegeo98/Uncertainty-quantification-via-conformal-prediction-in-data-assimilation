#!/usr/bin/env python
# -*- coding: utf-8 -*-
from parameters import *
from msw_model import sweq, initialize, noise_u
import assimilation as assim
import sys
import numpy as np
import random
import os.path
import os
import math
import numpy.linalg as lin
import numpy.ma as ma
import pickle
from tqdm import tqdm

CQR_CP_PATHS = [
    r"/home/mga5500/Schreibtisch/PYTHON/cqr_ensemble_latest.pkl"
]

def load_cp_scores(paths):
    scores = {}
    for p in paths:
        try:
            with open(p, "rb") as f:
                scores = pickle.load(f)
            print(f"[CQR-CP] Loaded conformal scores: {p}")
            break
        except Exception as e:
            print(f"[CQR-CP] Could not load {p}: {e}")
    if not scores:
        print("[CQR-CP] WARNING: No CP scores loaded. All CP=0.")
    return scores

conformal_scores_all = load_cp_scores(CQR_CP_PATHS)

def cp_for_cycle_cqr(scores, ti):
    entry = scores.get(ti)
    if entry is None:
        entry = scores.get(str(ti))
    if entry is None:
        return {'u': np.zeros(nx), 'h': np.zeros(nx), 'r': np.zeros(nx)}
    return {
        'u': np.asarray(entry.get('u', np.zeros(nx))),  # (nx,)
        'h': np.asarray(entry.get('h', np.zeros(nx))),  # (nx,)
        'r': np.asarray(entry.get('r', np.zeros(nx))),  # (nx,)
    }

def add_cp_random_interval_cqr(E, cp, seed=None):
    rng = np.random.default_rng(seed)
    nrows, k_ens = E.shape

    # cp['u'], cp['h'], cp['r'] are all (nx,) spatial vectors
    cp_vec = np.concatenate([
        np.asarray(cp['u']),   # (nx,)
        np.asarray(cp['h']),   # (nx,)
        np.asarray(cp['r']),   # (nx,)
    ])                         # (3*nx,)

    cp_mat = cp_vec[:, None]   # (3*nx, 1)
    N2 = rng.normal(0.0, 1.0, size=E.shape)
    perturb = (cp_mat * N2) / 2

    return E + perturb


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

print("METHODS:", methods)
if 'NN' in methods:
    from nn_assim import *

rd = 0

for seed in range(1, 11):
    print(f"\n RUNNING SEED {seed}")

    data = {}
    data['truth'] = {}
    data['obs_pos'] = {}

    for method in methods:
        data[method] = {}
        data[method]['analysis'] = {}
        data[method]['first guess'] = {}
        data[method]['analysis_cp'] = {}   # store CP-perturbed analyses

    if 'NN' in methods:
        data['NN']['lower'] = {}
        data['NN']['upper'] = {}

    if training:
        data['train'] = {}

    r = [random.Random(10000 + seed * 100000 + s * 1000 + rd) for s in range(99)]

    datadir_seed = (
        datadir
        + 'nsub' + str(nsub)
        + '/nu' + str(nu)
        + '/' + str(k)
        + '/CQR_enkf_latest/'
        + str(seed)
        + '/'
    )
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
            data[method]['first guess'][ti] = ens[method]

        obs, obs_original = assim.genobs(r[2], r[3], np.copy(truth))
        if radarint == 1:
            obs_position = assim.radar(np.copy(truth), r[4])
        data['obs_pos'][ti] = obs_position

        for method in methods:
            if method == 'QPEns' and training:
                ens[method], data['train'][ti] = assim.assimilation(
                    ens[method], obs_position, obs, method)
            else:
                ens[method] = assim.assimilation(
                    ens[method], obs_position, obs, method)

            if method == 'NN' and 'EnKF' in methods and ti >= 21:
                ens['NN'] = ens['EnKF'].copy()

            if method == 'NN' and ti >= 20:
                print('Start NN')
                ens['NN'], lower, upper = nn_assim(ens['NN'], obs_position)
                data['NN']['lower'][ti] = lower
                data['NN']['upper'][ti] = upper
                print('Stop NN')

            data[method]['analysis'][ti] = ens[method].copy()

            # CP Perturbation (from ti=20, only when EnKF present)
            if method == 'NN' and ti >= 20 and 'EnKF' in methods:
                nn_bg = np.copy(data['EnKF']['analysis'][ti])   # NN analysis at ti
                cp_t  = cp_for_cycle_cqr(conformal_scores_all, ti)

                nn_bg_cp = add_cp_random_interval_cqr(
                    nn_bg, cp_t, seed=ti)

                data['EnKF']['analysis_cp'][ti] = nn_bg_cp.copy()
                ens['EnKF'] = nn_bg_cp.copy()

    save_obj(data['truth'],   f'{datadir_seed}truth_data')
    save_obj(data['obs_pos'], f'{datadir_seed}obs_pos')
    for method in methods:
        save_obj(data[method], f'{datadir_seed}{method}_data')
    if training:
        save_obj(data['train'], f'{datadir_seed}train_data')
