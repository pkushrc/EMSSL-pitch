# coding=utf-8
import os
import glob
import pickle
import sys
import numpy as np
from tqdm import tqdm
import multiprocessing
import argparse

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator
from torch.nn.utils.rnn import pad_sequence

from config import cfg
from preprocess_audio_trm import audioProcess
from utils import mkdir_p, get_filename

############################ process audio and trm_params, make dataset #########################
def preprocess_raw_data_single(audio_fp, trm_params_fp, processed_fp, audio_processor):
    waveform, sample_rate = torchaudio.load(audio_fp) # waveform: tensor with shape[channel, length]
    spec = audio_processor.process_single_audio(audio_fp)
    tvs, track = audio_processor.process_single_trm_params(trm_params_fp)

    spec_num_frames = spec.shape[0]  # (frames, 80)
    tvs_num_frames = tvs.shape[1]  # (frames, 16)
    if tvs_num_frames >= spec_num_frames:
        tvs = tvs[:,:spec_num_frames, :]
    else:
        spec = spec[:tvs_num_frames, :]
    assert spec.shape[0] == tvs.shape[1]

    with open(processed_fp, 'wb') as fw:
        pickle.dump((waveform, sample_rate, spec, tvs, track), fw)

def preprocess_raw_data_single_batch(audio_fps, trm_params_fps, processed_fps, moments_path):
    '''
    HINT: 
        1. MUST have corresponding trm fps, audio_fps, and processed_fps
    '''
    audio_processor = audioProcess(moments_path)
    print('Preprocessing audios and trm_params...')
    for audio_fp, trm_fp, processed_fp in zip(audio_fps, trm_params_fps, processed_fps):
        preprocess_raw_data_single(audio_fp, trm_fp, processed_fp, audio_processor)

def preprocess_raw_data_multi_processes(audio_fps, trm_params_fps, processed_fps, moments_path, num_processes):
    '''
    HINT: 
        1. MUST have corresponding trm fps, audio_fps, and processed_fps
    '''
    # Run with multi-processes
    audio_processor = audioProcess(moments_path)
    print('Preprocessing audios and trm_params...')
    pool = multiprocessing.Pool(num_processes)
    for audio_fp, trm_fp, processed_fp in zip(audio_fps, trm_params_fps, processed_fps):
        pool.apply_async(preprocess_raw_data_single, (audio_fp, trm_fp, processed_fp, audio_processor))
    pool.close() 
    pool.join() 


class A2TDataset(Dataset):
    '''
    Dataset with preprocessed audios and trm_params as input: processed_fps
    '''
    def __init__(self, data_fps, drop_r0=True):
        '''
            since r0 of tvs always have same value, no need to train it
        '''
        super(Dataset, self).__init__()
        self.processed_fps = data_fps
        self.drop_r0 = drop_r0

    def __getitem__(self, index):
        with open(self.processed_fps[index], 'rb') as fr:
            waveform, sample_rate, spec, tvs, track = pickle.load(fr)
        seq_len = spec.shape[0]
        
        if self.drop_r0:
            tvs = np.concatenate((tvs[:,:, :cfg.AUDIO.TVS_R0_DIM], tvs[:, :,cfg.AUDIO.TVS_R0_DIM + 1:]), axis=-1)

        return waveform[0], torch.tensor(sample_rate, dtype=torch.int), torch.tensor(spec, dtype=torch.float32), \
                torch.tensor(tvs, dtype=torch.float32), torch.tensor(track, dtype=torch.float32), torch.tensor(seq_len)

    def __len__(self):
        return len(self.processed_fps)


class A2TDatasetFromRaw(Dataset):
    '''
    Dataset with non-preprocessed audios and trm_parameters as input.
    Works well when model is run on cpu, but if use gpu, better preprocess data first.
    '''
    def __init__(self, audio_fps, trm_params_fps, moments_path, drop_r0):
        '''
            since r0 of tvs always have same value, no need to train it
        '''
        super(Dataset, self).__init__()
        self.audio_processor = audioProcess(moments_path)
        self.audio_fps = audio_fps
        self.trm_params_fps = trm_params_fps
        self.drop_r0 = drop_r0

    def __getitem__(self, index):
        waveform, sample_rate = torchaudio.load(self.audio_fps[index])
        spec = self.audio_processor.process_single_audio(self.audio_fps[index])
        tvs, track = self.audio_processor.process_single_trm_params(self.trm_params_fps[index])
        spec_num_frames = spec.shape[0]
        tvs_num_frames = tvs.shape[1]
        if tvs_num_frames >= spec_num_frames:
            tvs = tvs[:,:spec_num_frames, :]
        else:
            spec = spec[:tvs_num_frames, :]
            assert spec.shape[0] == tvs.shape[1]
        seq_len = spec.shape[0]

        if self.drop_r0:
            tvs = np.concatenate((tvs[:,:, :cfg.AUDIO.TVS_R0_DIM], tvs[:,:, cfg.AUDIO.TVS_R0_DIM + 1:]), axis=-1)

        return waveform[0], torch.tensor(sample_rate, dtype=torch.int), torch.tensor(spec, dtype=torch.float32), \
                torch.tensor(tvs, dtype=torch.float32), torch.tensor(track, dtype=torch.float32), torch.tensor(seq_len)

    def __len__(self):
        return len(self.audio_fps)

def collate_fn(batch):
    """
    Manually build a batch from a list to a tensor
    :param batch: a list of tensors
    :return: 
        list_wav: a list of wav with length B , [wav1, wav2, wav3, ...]
        list_fr: a list of sample rates with length B, [sr1, sr2, sr3, ...]
        batch_spec: a tensor with shape of (B, max_len, num_fbank, 1)
        batch_tvs: a tensor with shape of (B, max_len, num_tvs_dim)
        batch_track: a tensor with shape of (B, num_track_dim)
        batch_seq_len: a tensor with shape of (B, )
    """
    list_wav, list_sr, list_spec, list_tvs, list_track, list_seq_len = map(list, zip(*batch))
    batch_wav_len = torch.tensor([waveform.shape[0] for waveform in list_wav])
    batch_wav = pad_sequence(list_wav, batch_first=True)
    batch_sr = torch.stack(list_sr)
    batch_spec = pad_sequence(list_spec, batch_first=True).unsqueeze(-1)

    list_tvs_tmp = [x.permute(1,0,2)  for x in list_tvs]
    batch_tvs = pad_sequence(list_tvs_tmp, batch_first=True)
    batch_tvs = batch_tvs.permute(0,2,1,3)
    batch_track = torch.stack(list_track)
    batch_seq_len = torch.stack(list_seq_len)
    return batch_wav, batch_wav_len, batch_sr, batch_spec, batch_tvs, batch_track, batch_seq_len


########################### process audio and make dataset################
def audio_preprocess_single(audio_fp, processed_fp, audio_processor):
    spec = audio_processor.process_single_audio(audio_fp)
    with open(processed_fp, 'wb') as fw:
        pickle.dump(spec, fw)

def audio_preprocess_single_batch(audio_fps, processed_fps, moments_path):
    audio_processor = audioProcess(moments_path)
    print('Using moments: {}'.format(moments_path))
    print(audio_processor.kaldi_params)
    for audio_fp, processed_fp in zip(audio_fps, processed_fps):
        audio_preprocess_single(audio_fp, processed_fp, audio_processor)

def audio_preprocess_multi_processes(audio_fps, processed_fps, moments_path, num_processes):
    # Run with multi-processes
    print('Preprocessing audios...')
    audio_processor = audioProcess(moments_path)
    pool = multiprocessing.Pool(num_processes)
    for audio_fp, processed_fp in zip(audio_fps, processed_fps):
        pool.apply_async(audio_preprocess_single, (audio_fp, processed_fp, audio_processor))
    pool.close() 
    pool.join() 


class AudioDataset(Dataset):
    '''
    Dataset with preprocessed audios as input.
    '''
    def __init__(self, data_fps):
        super(Dataset, self).__init__()
        self.processed_fps = data_fps

    def __getitem__(self, index):
        with open(self.processed_fps[index], 'rb') as fr:
            spec = pickle.load(fr)
        seq_len = spec.shape[0]

        return torch.tensor(spec, dtype=torch.float32), torch.tensor(seq_len)

    def __len__(self):
        return len(self.processed_fps)


class AudioDatasetFromRaw(Dataset):
    '''
    Dataset with non-preprocessed audios as input.
    Works well when model is run on cpu, but if use gpu, better preprocess data first
    '''
    def __init__(self, data_fps, moments_path):
        super(Dataset, self).__init__()
        self.audio_processor = audioProcess(moments_path)
        self.audio_fps = data_fps

    def __getitem__(self, index):
        spec = self.audio_processor.process_single_audio(self.audio_fps[index])
        seq_len = spec.shape[0]

        return torch.tensor(spec, dtype=torch.float32), torch.tensor(seq_len)

    def __len__(self):
        return len(self.audio_fps)

def audio_collate_fn(batch):
    """
    Manually build a batch from a list to a tensor
    :param batch: a list of tensors
    :return: 
        batch_spec: a tensor with shape of (B, max_len, num_fbank, 1)
        batch_seq_len: a tensor with shape of (B, )
    """
    list_spec, list_seq_len = map(list, zip(*batch))
    batch_spec = pad_sequence(list_spec, batch_first=True).unsqueeze(-1)
    batch_seq_len = torch.stack(list_seq_len)
    return batch_spec, batch_seq_len

########################### data loader #######################
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class DataPrefetcher():
    '''
    Preload to cuda stream
    '''
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                if k != 'meta':
                    self.batch[k] = self.batch[k].cuda(non_blocking=True)

    def __iter__(self):
        return self

    def __next__(self):
        if not self.batch == None:
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = self.batch
            self.preload()
            return batch
        else:
            raise StopIteration


######################### Preprocess data in given folders ###########################
def prepare_raw_dataset(moments_path, source_dir, target_dir):
    '''
    Preprocessing audios given in source_dir
    '''
    audio_fps = glob.glob(os.path.join(source_dir, '*.wav'))
    processed_fps = [fp.replace('.wav', '_spec.pkl').replace(source_dir, target_dir) for fp in audio_fps]
    for fp in processed_fps:
        mkdir_p(fp.strip(get_filename(fp)))
    print('{} audios from {} to be processed'.format(len(audio_fps), source_dir))
    audio_preprocess_single_batch(audio_fps, processed_fps, moments_path=moments_path)
    print('Prepared data!')

def prepare_A2T_dataset(moments_path, source_dir):
    '''
    Given directory with audios and corresponding trm_parameters, processing the files.
    Processed files will be saved in the source_dir
    '''
    audio_fps = glob.glob(os.path.join(source_dir, '*.wav'))
    trm_params_fps = [fp.replace('wav_output', 'trm_param').replace('.wav', '.txt') for fp in audio_fps]
    processed_fps = [fp.replace('wav_output', 'spec_tvs_track').replace('.wav', '.pkl') for fp in audio_fps]
    print('{} audios from {} to be processed'.format(len(audio_fps), source_dir))
    preprocess_raw_data_single_batch(audio_fps, trm_params_fps, processed_fps, moments_path)
    print('Prepared data!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--moments_path', type=str)
    parser.add_argument('--source_dir', type=str)
    parser.add_argument('--target_dir', type=str)
    args = parser.parse_args()
    prepare_raw_dataset(args.moments_path, args.source_dir, args.target_dir)
