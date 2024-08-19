#coding=utf-8
import os
import glob
import sys
import re
import argparse
import pickle

sys.path.append('..')
sys.path.append('../misc/')
sys.path.append('../trainer/')
from a2t_model_util import A2TUtil
from misc.utils import get_filename
from misc.config import cfg
from misc.spec_evaluation import eval_sent_snr
from a2t_model_util import extract_synthesize_preprocess_from_ori_spec

def model_test_data_prepare(source_audio_dir, data_save_dir, moments_path, gnuspeech_path):
    '''
    Prepare test data for model testing, will doing the following processes:
    Extract trm --> generate audio --> process generated audio, saved in data_save_dir.
    ***** Model should be specified in cfg.UTIL
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(','.join([str(idx) for idx in cfg.UTIL.GPU_ID]))
    # raw_audio_fps = sorted(glob.glob(os.path.join(source_audio_dir, '*/test/*_spec.pkl'))) # should have been preprocessed
    ref_spec_fps = sorted(glob.glob(os.path.join(source_audio_dir, '*_spec.pkl')))
    print('{} reference spectrograms from: {}'.format(len(ref_spec_fps), source_audio_dir))
    A2TModel = A2TUtil(moments_path, model=None) # load model specified in cfg.UTIL
    extract_synthesize_preprocess_from_ori_spec(A2TModel, gnuspeech_path, moments_path, ref_spec_fps, source_audio_dir, data_save_dir)

def evaluate_model_with_sentence_SNR(source_audio_dir, data_save_dir, moments_path, gnuspeech_path):
    '''
    Prepare test data and evaluate sentence SNR of synthetic audios.
    ***** Model should be specified in cfg.UTIL
    '''
    model_test_data_prepare(source_audio_dir, data_save_dir, moments_path, gnuspeech_path)
    spec_snr_mean, spec_snr_std = eval_sent_snr(source_audio_dir, data_save_dir)
    return spec_snr_mean, spec_snr_std

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_audio_fp', type=str)
    parser.add_argument('--source_audio_dir', type=str)
    parser.add_argument('--data_save_dir', type=str)
    parser.add_argument('--test_result_fp', type=str)
    parser.add_argument('--mode', type=str) # extract_single, extract_batch, model_test
    parser.set_defaults(
        raw_audio_fp='',
        source_audio_dir='',
        data_save_dir='',
        test_result_fp='',
        mode='model_test') 
    args = parser.parse_args()
    
    if args.mode == 'extract_single':
        trained_model = A2TUtil(cfg.GLOBAL.MOMENTS_PATH) # model specified in cfg.UTIL
        trained_model.extract_and_save_articulatory_info(args.raw_audio_fp, args.trm_params_out_fp)
    elif args.mode == 'extract_batch':
        trained_model = A2TUtil(cfg.GLOBAL.MOMENTS_PATH) # model specified in cfg.UTIL
        audio_fps = glob.glob(os.path.join(args.source_audio_dir, '*.wav'))
        trm_params_fps = [os.path.join(args.trm_params_out_dir, get_filename(fp).strip('.wav').strip('_') + \
                                                                    '_trm_params.txt') for fp in audio_fps]
        print('Extrating trm_params of {} audios...'.format(len(audio_fps)))
        trained_model.extract_and_save_articulatory_info(audio_fps, trm_params_fps, processed_audio=False)
    elif args.mode == 'model_test': # test model with sentence SNR
        spec_snr_mean, spec_snr_std = evaluate_model_with_sentence_SNR(args.source_audio_dir, \
                                        args.data_save_dir, cfg.GLOBAL.MOMENTS_PATH, cfg.GLOBAL.GNUSPEECH_DIR)
        with open(args.test_result_fp, 'wb') as fw:
            pickle.dump((spec_snr_mean, spec_snr_std), fw)
        print('Sentence SNR results on {}, mean: {}, std: {}'.format(args.source_audio_dir, spec_snr_mean, spec_snr_std))
    else:
        raise RuntimeError('Not implemented util mode!')

