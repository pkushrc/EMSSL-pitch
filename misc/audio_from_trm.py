#coding=utf-8
# ***************
# Generate audios from trm parameters
# ***************
import sys
import os
import glob
from tqdm import tqdm
import numpy as np
import multiprocessing
import soundfile as sf
def generate_audio_single(trm_params_fp, audio_fp, generator_path):
    '''
    Call gunspeech_sa to generate a piece of audio from trm parameters, run with single process
    '''
    audio_fp1 = audio_fp.replace('.wav','s1.wav')
    audio_fp2 = audio_fp.replace('.wav','s2.wav')
    os.system(os.path.join(generator_path, 'gnuspeech_sa') + ' -t {} -o {}'.format(trm_params_fp, audio_fp1))
    wav1 ,sr1 = sf.read(audio_fp1)
    os.system(os.path.join(generator_path, 'gnuspeech_sa') + ' -t {} -o {}'.format(trm_params_fp.replace('s1','s2'), audio_fp2))
    wav2 ,sr2 = sf.read(audio_fp2)
    length = max(wav1.shape[0],wav2.shape[0])
    if wav1.shape[0]>wav2.shape[0]:
        wav1[0:wav2.shape[0]]  = wav1[0:wav2.shape[0]] + wav2
        sf.write(audio_fp,wav1,sr1)
    else:
        wav2[0:wav1.shape[0]]  = wav2[0:wav1.shape[0]] + wav1
        sf.write(audio_fp,wav2,sr1)
    

   
def generate_audio_multi_processes(trm_params_fps, audio_fps, generator_path, num_processes):
    '''
    Generate a bounch of audio from trm parameters, run with MULTI process
    '''
    pool = multiprocessing.Pool(num_processes)
    for trm_fp, audio_fp in zip(trm_params_fps, audio_fps):
        pool.apply_async(generate_audio_single, (trm_fp, audio_fp, generator_path))
    pool.close() 
    pool.join() 

if __name__ == "__main__":
    gnuspeech_dir = 'gunspeech_sa/build'
    trm_params_fp = 'trm_dir/trm_params.txt'
    audio_gen_fp = 'trm_dir/wav_output.wav'

    generate_audio_single(trm_params_fp, audio_gen_fp, generator_path=gnuspeech_dir)
    