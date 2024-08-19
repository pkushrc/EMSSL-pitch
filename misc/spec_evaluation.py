#coding=utf-8
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import cm
import multiprocessing

_TINY = 1e-8

def rescale_spec(spec):
    '''
    Make sure value in spec positive and have scale in [0, 1]
    Because we need spec to represent energy and have a straight understanding of the number
    '''
    spec = (spec + 1) / 2 # determined by the normalize method define in proprocess_audio_trm.py
    return spec

def prepare_spec(ori_spec_fp, gen_spec_fp):
    '''
    Rescale and truncate specs for evaluation
    Input:
        ori_spec_fp: preprocessed spectrogram of original audio
        gen_spec_fp: preprocessed spectrogram of original audio, with processed trm parameters and audio
    Output:
        spec_1, spec_2: [seq_length, fbank_num], rescaled
    '''
    # Load spec
    with open(ori_spec_fp, 'rb') as fr_ori:
        spec_1 = pickle.load(fr_ori)
    with open(gen_spec_fp, 'rb') as fr_gen:
        _, _, spec_2, _, _ = pickle.load(fr_gen)

    # Check length and cut off
    assert(abs(spec_1.shape[0] - spec_2.shape[0]) <= 5)
    # print(spec_1.shape[0], spec_2.shape[0])
    len_spec = min(spec_1.shape[0], spec_2.shape[0])
    spec_1 = spec_1[:len_spec, :]
    spec_2 = spec_2[:len_spec, :]

    # # Make sure value in spec positive and have scale in [0, 1]
    spec_1 = rescale_spec(spec_1)
    spec_2 = rescale_spec(spec_2)

    return spec_1, spec_2

################### local compare ###################
def get_local_diff(spec_pair_list):
    spec_ori_list, spec_gen_list = zip(*spec_pair_list)
    spec_ori = np.concatenate(spec_ori_list, axis=0)
    spec_gen = np.concatenate(spec_gen_list, axis=0)
    local_diff = np.abs(spec_ori - spec_gen)
    return local_diff

def get_local_snr(spec_pair_list):
    spec_ori_list, spec_gen_list = zip(*spec_pair_list)
    spec_ori = np.concatenate(spec_ori_list, axis=0)
    spec_gen = np.concatenate(spec_gen_list, axis=0)
    local_snr = 20 * np.log10((spec_ori + _TINY)/(np.abs(spec_ori - spec_gen + _TINY)))
    return local_snr

def eval_local_snr_and_save(ori_pkl_dir, gen_pkl_dir, local_snr_save_fp=None):
    '''
    Return mean/std of all t-f bins or mean/std of every single bin
    '''
    start_ = time.time()
    gen_audio_fps = sorted(glob.glob(os.path.join(gen_pkl_dir, '*.pkl')))
    ori_audio_fps = [fp.replace(gen_pkl_dir, ori_pkl_dir).replace('_spec_tvs_track.pkl', '_spec.pkl') for fp in gen_audio_fps]

    spec_pair_list = [prepare_spec(ori_fp, gen_fp) for ori_fp, gen_fp in zip(ori_audio_fps, gen_audio_fps)]
    spec_local_snr = get_local_snr(spec_pair_list)
    # print('shape of local snr: {}'.format(spec_local_snr.shape))
    # if local_snr_save_fp: np.savetxt(local_snr_save_fp, spec_local_snr, delimiter = ',') 
    if local_snr_save_fp:
        with open(local_snr_save_fp, 'wb') as fw:
            pickle.dump(spec_local_snr, fw)

    spec_snr_mean_bin = np.mean(spec_local_snr, axis=0)
    spec_snr_mean_all = np.mean(spec_snr_mean_bin)

    spec_snr_std_bin = np.std(spec_local_snr, axis=0)
    spec_snr_std_all = np.std(spec_local_snr)

    end_ = time.time()
    print('{:4f}s used to eval spec from {}'.format(end_ - start_, gen_pkl_dir))

    return spec_snr_mean_all, spec_snr_mean_bin, spec_snr_std_all, spec_snr_std_bin

############### sentence level compare ############
def get_sentence_snr_single(spec_ori, spec_gen):
    spec_noise = np.abs(spec_ori - spec_gen)
    sentence_snr = 10 * np.log10(np.sum(spec_ori ** 2) / (np.sum(spec_noise ** 2) + _TINY))
    return sentence_snr

def sent_snr_list(ori_pkl_dir, gen_pkl_dir, return_fps=False):
    '''
    Return a list of sentence snr
    '''
    gen_audio_fps = sorted(glob.glob(os.path.join(gen_pkl_dir, '*.pkl')))
    ori_audio_fps = [fp.replace(gen_pkl_dir, ori_pkl_dir).replace('_spec_tvs_track.pkl', '_spec.pkl') for fp in gen_audio_fps]

    spec_pair_list = [prepare_spec(ori_fp, gen_fp) for ori_fp, gen_fp in zip(ori_audio_fps, gen_audio_fps)]
    spec_snr_list = [get_sentence_snr_single(spec_ori, spec_gen) for spec_ori, spec_gen in spec_pair_list]
    
    if return_fps:
        return spec_snr_list, ori_audio_fps, gen_audio_fps
    else:
        return spec_snr_list

def eval_sent_snr(ori_pkl_dir, gen_pkl_dir, iter_label=None):
    '''
    Return mean and std of sentences snr, can be used independently 
    '''
    start_ = time.time()
    spec_snr = sent_snr_list(ori_pkl_dir, gen_pkl_dir)
    spec_snr_mean = np.mean(spec_snr)
    spec_snr_std = np.std(spec_snr)
    end_ = time.time()
    print('{:4f}s used to eval spec from {}'.format(end_ - start_, gen_pkl_dir))

    if iter_label:
        return iter_label, spec_snr_mean, spec_snr_std
    else:
        return spec_snr_mean, spec_snr_std

################ Do evaluation experiments #############
def do_evaluation(ori_pkl_dir, gen_pkl_iter_dir, iter_num, num_processes=2):
    print('Start eval spec...')
    start_ = time.time()
    results = []
    pool = multiprocessing.Pool(num_processes)
    for i in iter_num:
        results.append(pool.apply_async(eval_sent_snr, (ori_pkl_dir, gen_pkl_iter_dir.format(i), i)))
    pool.close() 
    pool.join() # wait until all processes in pool are done, within the pool, there is no order
    end_ = time.time()
    print('{:4f}s used to eval specs from {} iterations.'.format(end_ - start_, len(iter_num)))

    # get results and sort
    collect = dict()
    snr_collect = []
    for r in results:
        iter_s, spec_snr_mean, spec_snr_std = r.get()
        collect[iter_s] = [spec_snr_mean, spec_snr_std]
    for i in iter_num:
        snr_collect.append(collect[i])
    spec_snr_mean_l, spec_snr_std_l = zip(*snr_collect)

    return iter_num, spec_snr_mean_l, spec_snr_std_l

def eval_sent_snr_multi_person_single_iter(ori_pkl_dir, gen_pkl_dir, p_list, iter_label):
    '''
    Evaluation of a single iteration but with multi persons;
    Return both mean/std of sentences snr for all or every single person
    '''
    start_ = time.time()

    snr_mean_p_list = []
    snr_std_p_list = []
    snr_all_collect = []
    for p in p_list:
        spec_snr_list = sent_snr_list(ori_pkl_dir.format(p), gen_pkl_dir.format(iter_label, p))
        snr_mean_p_list.append(np.mean(spec_snr_list))
        snr_std_p_list.append(np.std(spec_snr_list))
        snr_all_collect.extend(spec_snr_list)
    snr_mean_all = np.mean(snr_all_collect)
    snr_std_all = np.std(snr_all_collect)

    end_ = time.time()
    print('{:4f}s used to eval spec from {}'.format(end_ - start_, gen_pkl_dir.format(iter_label, 'all')))
    return iter_label, snr_mean_all, snr_std_all, snr_mean_p_list, snr_std_p_list

def do_evaluation_snr_multi_person(ori_pkl_dir, gen_pkl_dir, p_list, iter_num, num_processes=2):
    print('Start eval spec...')
    start_ = time.time()
    results = []
    pool = multiprocessing.Pool(num_processes)
    for i in iter_num:
        results.append(pool.apply_async(eval_sent_snr_multi_person_single_iter, (ori_pkl_dir, gen_pkl_dir, p_list, i)))
    pool.close() 
    pool.join() # wait until all processes in pool are done, within the pool, there is no order
    end_ = time.time()
    print('{:4f}s used to eval specs from {} iterations.'.format(end_ - start_, len(iter_num)))

    # get results and sort
    collect = dict()
    snr_collect = []
    for r in results:
        iter_label, snr_mean_all, snr_std_all, snr_mean_p_list, snr_std_p_list = r.get()
        collect[iter_label] = [snr_mean_all, snr_std_all, snr_mean_p_list, snr_std_p_list]
    for i in iter_num:
        snr_collect.append(collect[i])
    snr_mean_all_l, snr_std_all_l, snr_mean_p_list_l, snr_std_p_list_l = zip(*snr_collect)

    return iter_num, snr_mean_all_l, snr_std_all_l, snr_mean_p_list_l, snr_std_p_list_l

##################### plot function #####################
def plot_spec_snr_mean_std(iter_l, snr_mean_l, snr_std_l):
    plt.errorbar(iter_l, snr_mean_l, yerr=snr_std_l, fmt='o', color='b', ecolor='r', elinewidth=1, capsize=4)
    plt.xlabel('iteration')
    plt.ylabel('mean local snr')
    plt.show()

def plot_spec_snr_bin(snr_mean_bin, snr_std_bin):
    plt.errorbar(np.arange(1, len(snr_mean_bin) + 1, step=1), snr_mean_bin, yerr=snr_std_bin, fmt='o', color='b', ecolor='r', elinewidth=1, capsize=4)
    plt.xlabel('bin')
    plt.ylabel('mean local snr')
    plt.show()


if __name__ == "__main__":
    ori_audio_dir = 'processed_audio_dir/test'
    gen_audio_dir = 'save_dir/test'
    result_txt = './eval_result.pkl'

    ############## score sentence SNR for several iterations for a single person ##########
    iter_num = np.arange(1, 100, step=1)
    iter_num, spec_snr_mean_l, spec_snr_std_l = do_evaluation(ori_audio_dir, gen_audio_dir, iter_num, num_processes=8)
    with open(result_txt, 'wb') as fw:
        pickle.dump((iter_num, spec_snr_mean_l, spec_snr_std_l), fw)
    with open(result_txt, 'rb') as fr:
        iter_num, spec_snr_mean_l, spec_snr_std_l = pickle.load(fr)
    plot_spec_snr_mean_std(iter_num, spec_snr_mean_l, spec_snr_std_l)
