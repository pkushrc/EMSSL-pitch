# coding=utf-8
# ***************
# Doing statistics of tvs and spec for normalization
# ***************

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import argparse

from preprocess_audio_trm import audioProcess
from config import cfg

def draw_hist(values, name):
    plt.hist(values, 100)
    plt.xlabel('value')
    plt.ylabel('Count')
    plt.title('Count of {}'.format(name))
    plt.show()

def count_tvs(trm_params_fps, save_path):
    print('Doing statistics on {} files'.format(len(trm_params_fps)))
    audio_processor = audioProcess()
    tvs = [audio_processor.extract_trm_parameters(fp)[0] for fp in trm_params_fps]
    print('Get {} tvs'.format(len(tvs)))
    tvs = np.concatenate(tvs, axis=0)
    tvs_labels = cfg.AUDIO.TVS_DIM_NAME
    with open(save_path, 'w', encoding='utf-8') as f_w:
        for i in range(cfg.AUDIO.TVS_DIM):
            hists, edges = np.histogram(tvs[:, i], bins=60, density=True)
            f_w.writelines('statistics of {}, min: {:.3f}, max: {:.3f}\n'.format(tvs_labels[i], min(tvs[:, i]), max(tvs[:, i])))
            f_w.writelines('hists: ' + ' '.join(['{:.3f}'.format(h) for h in hists]) + '\n')
            f_w.writelines('edges: ' + ' '.join(['{:.3f}'.format(e) for e in edges]) + '\n')

def make_moments(moments_path):
    '''
    Save to moments file.
    '''
    _spec_min = -16
    _spec_max = 10

    _tvs_min = np.array([-30.0, 0.0, 0.0, 0.0, 0.0, 500.0, 500.0, 0.8, 0.2, 0.2, 0.2, 0.0, 0.0, 0.2, 0.0, 0.0], dtype=np.float32)
    _tvs_max = np.array([17.0, 60.0, 20.0, 1.5, 7.0, 7000.0, 5000.0, 0.8, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 3.0, 1.0], dtype=np.float32)
    
    _track_min = np.array([55, 8, -24, 30, 10, 15, 0.5, 0.2, 2, 2000, 2000, 0.5, 1200], dtype=np.float32)
    _track_max = np.array([60, 20, 20, 45, 35 ,40, 8, 5, 10, 6000, 6000, 10, 1800], dtype=np.float32)

    with open(moments_path, 'wb') as f_w:
        pickle.dump((_spec_min, _spec_max, _tvs_min, _tvs_max, _track_min, _track_max), f_w)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--moments_path', type=str)
    parser.set_defaults(
        moments_path='') 
    args = parser.parse_args()
    
    make_moments(args.moments_path)




    
    
