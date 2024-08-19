#coding = utf-8
import os
import time
import numpy as np
import math
import torchaudio
from six.moves import xrange
import pickle
from scipy.interpolate import interp1d

from config import cfg

_TINY = 1e-8

class audioProcess:
    def __init__(self, moments_path=None):
        self.moments_path = moments_path
        if moments_path:
            with open(moments_path, 'rb') as f:
                self._spec_min, self._spec_max, self._tvs_min, self._tvs_max, self._track_min, self._track_max = pickle.load(f)
        self.kaldi_params = {"channel": -1,
                             "dither": cfg.AUDIO.DITCHER, # to add some noise to silent: 0.00001
                             "frame_length": cfg.AUDIO.FRAME_LENGTH,
                             "frame_shift": cfg.AUDIO.FRAME_SHIFT,
                             "high_freq": cfg.AUDIO.HIGH_F,
                             "low_freq": cfg.AUDIO.LOW_F,
                             "num_mel_bins": cfg.AUDIO.FBANK_NUM,
                             "preemphasis_coefficient": 0.97,
                             "remove_dc_offset": True, #Subtract mean from waveform on each frame
                             "use_energy": False, #add an extra dimension with energy to the FBANK output
                             "round_to_power_of_two": True, #If True, round window size to power of two by zero-padding input to FFT.
                             "use_log_fbank": True, # If true, produce log-filterbank, else produce linear.
                             "use_power": True, #If true, use power, else use magnitude
                             "window_type": 'hamming', #‘hamming’|’hanning’|’povey’|’rectangular’|’blackman’
                             }

    ####################################################################
    # Following are audio process methods

    # def get_frame_num(self, signal_length, frame_length, frame_step):
    #     if signal_length <= frame_length: 
    #         frame_num = 1
    #     else:
    #         frame_num = 1 + int(math.ceil((1.0 * signal_length - frame_length) / frame_step))
    #     return frame_num

    def normalize_spec(self, spec):
        spec_norm = (spec - self._spec_min) / (self._spec_max - self._spec_min + _TINY)
        spec_norm = spec_norm * 2 - 1 # rerange to [-1, 1]
        spec_norm = np.clip(spec_norm, -1., 1.)
        return spec_norm
    
    def process_single_audio(self, audio_fp, self_normalize=False):
        '''
        process a single audio to normalized spectrum
        input:
            audio_fp: audio file path
            self_normalize: deprecated, use statistics of the spec to normalize itself, 
                            need spec_mean and spec_std be none
        return: normalized spectrum of audio, [seq_length, num_fbank]
        '''
        waveform, sample_rate = torchaudio.load(audio_fp)
        spec = torchaudio.compliance.kaldi.fbank(waveform, 
                                                 sample_frequency=sample_rate, 
                                                 **self.kaldi_params).numpy() # [seq_length, mel_bins]
        spec = self.normalize_spec(spec)
        return spec

    ####################################################################
    # Following are tvs, track process methods
    def insert_r0(self, tvs, dropped_r0):
        '''
        During training, may drop r0 of tvs, since this dimension always have a same value.
        Then after model extrating tvs from audio, r0 need to be inserted.
        Input:
            tvs: np.2darray, [seq_length, tvs_dim]
            dropped_r0: whether the r0 dim in the input tvs have been dropped, if True, will insert
        Hint: this method apply to normalized tvs, should be used before unnormalize 
        '''
        if dropped_r0:
            assert(tvs.shape[-1] == cfg.AUDIO.TVS_DIM - 1)
            if len(tvs.shape) == 2:
                r0_vec = np.zeros((tvs.shape[0], 1)) + cfg.AUDIO.TVS_R0_VALUE_NORM
                tvs = np.concatenate((tvs[:, :cfg.AUDIO.TVS_R0_DIM], r0_vec, tvs[:, cfg.AUDIO.TVS_R0_DIM:]), axis=-1)
            elif len(tvs.shape) == 3:
                r0_vec = np.zeros((tvs.shape[0], tvs.shape[1], 1)) + cfg.AUDIO.TVS_R0_VALUE_NORM
                tvs = np.concatenate((tvs[:, :, :cfg.AUDIO.TVS_R0_DIM], r0_vec, tvs[:, :, cfg.AUDIO.TVS_R0_DIM:]), axis=-1)
            elif len(tvs.shape) == 4:
                r0_vec = np.zeros((tvs.shape[0], tvs.shape[1], tvs.shape[2],1)) + cfg.AUDIO.TVS_R0_VALUE_NORM
                tvs = np.concatenate((tvs[:, :, :, :cfg.AUDIO.TVS_R0_DIM], r0_vec, tvs[:, :, :, cfg.AUDIO.TVS_R0_DIM:]), axis=-1)
            else:
                raise NotImplementedError('insert_r0 only support 2d or 3d input')
        else:
            pass
        return tvs

    def interpolate_tvs(self, tvs, sample_rate, kind='linear'):
        '''
        sample tvs at time stamps of each spec frame
        input:
            tvs: original tvs, np.2darray [seq_length, tvs_dim] or np.3darray [bz, seq_length, tvs_dim]
            sample_rate: expect_rate / ori_rate
            kind: 'linear', 'quadratic'
        return: interpolated tvs, [seq_length, tvs_dim] or [bz, seq_length, tvs_dim]
        '''
        seq_length = tvs.shape[-2]
        x = np.linspace(0, seq_length - 1, seq_length)
        sample_f = interp1d(x, tvs, kind=kind, axis=-2)
        sample_x = np.arange(0, seq_length - 1, step=sample_rate)
        return sample_f(sample_x)

    def normalize_tvs(self, tvs):
        '''
        Support a single tvs with shape [seq_length, tvs_dim] 
             or a batch of tvs with shape [bz, seq_length, tvs_dim]
        '''
        tvs = np.clip(tvs, self._tvs_min, self._tvs_max)
        tvs_norm = (tvs - self._tvs_min) / (self._tvs_max - self._tvs_min + _TINY)
        tvs_norm = tvs_norm * 2 - 1 # rerange to [-1, 1]
        return tvs_norm

    def unnormalize_tvs(self, normalized_tvs):
        '''
        Support a single tvs with shape [seq_length, tvs_dim] 
             or a batch of tvs with shape [bz, seq_length, tvs_dim]
        '''
        tvs = (normalized_tvs + 1) / 2
        tvs = tvs * (self._tvs_max - self._tvs_min) + self._tvs_min
        return tvs

    def normalize_track(self, track):
        '''
        Support a single track with shape [track_dim] 
             or a batch of track with shape [bz, track_dim]
        '''
        track_norm = (track - self._track_min) / (self._track_max - self._track_min + _TINY)  # broadcast
        track_norm = track_norm * 2 - 1
        track_norm = np.clip(track_norm, -1., 1.)
        return track_norm

    def unnormalize_track(self, normalized_track):
        '''
        Support a single track with shape [track_dim] 
             or a batch of track with shape [bz, track_dim]
        '''
        track = (normalized_track + 1) / 2
        track = track * (self._track_max - self._track_min) + self._track_min
        return track

    def extract_trm_parameters(self, parameters_file):
        '''
        extract track struture and tvs from the parameters file
        input: filename
        output:
            tvs: 2-d array, [cfg.AUDIO.WINDOW_LEN, cfg.AUDIO.TVS_DIM]
            *** track_params: 1-d array, [cfg.AUDIO.TRACK_params_DIM]
        '''
        selected_param_idx = cfg.AUDIO.TRACK_IDX
        with open(parameters_file, 'r', encoding = 'utf-8') as f_r:
            lines = f_r.readlines()
            # Extract track_params
            track_params = [float(lines[idx].strip()) for idx in selected_param_idx]
            # track_params.append(get_mean_pitch(parameters_file)) # temporarily
            track_params = np.array(track_params, dtype = np.float32)
            
            # Extract tvs, need to sample later
            tvs_lines = lines[cfg.AUDIO.TVS_START:]
            tvs = np.array([line.strip().split(' ') for line in tvs_lines], dtype = np.float32)

            return tvs, track_params

    def process_single_trm_params(self, trm_params_fp):
        '''
        process trm parameter file to normalized and sampled  tvs and track
        input:
            trm_params_fp: trm parameter file path
            tvs_mean, tvs_std, track_mean, track_std: for normalization
        return: normalized tvs and track, tvs will have same time step of spec
        '''
        tvs, track = self.extract_trm_parameters(trm_params_fp)
        tvs = self.normalize_tvs(tvs)
        # Make sure time stamps of tvs and spec match
        tvs = self.interpolate_tvs(tvs, sample_rate=cfg.AUDIO.FRAME_SHIFT / 1000. * cfg.AUDIO.INPUT_CONTROL_RATE)
        track = self.normalize_track(track)
        tvs1 = tvs[np.newaxis,:,:]
        track1 = track[np.newaxis,:]

        tvs2, track2 = self.extract_trm_parameters(trm_params_fp.replace('s1','s2'))
        tvs2 = self.normalize_tvs(tvs2)
        # Make sure time stamps of tvs and spec match
        tvs2 = self.interpolate_tvs(tvs2, sample_rate=cfg.AUDIO.FRAME_SHIFT / 1000. * cfg.AUDIO.INPUT_CONTROL_RATE)
        track2 = self.normalize_track(track2)
        tvs2 = tvs2[np.newaxis,:,:]
        track2 = track2[np.newaxis,:]
        tvs = np.concatenate((tvs1,tvs2),axis = 0)
        track = np.concatenate((track1,track2),axis = 0)

        return tvs, track

    def save_trm_params(self, tvs, track, save_path, pad=True):
        '''
        Simply save processed track and tvs
        Input:
            pad: whether should add additional static track parameters, 
                 if track directly comes from a2t model, then should set pad=True  
        '''
        with open(save_path, 'w', encoding='utf8') as f_w:
            track_full = []
            if pad:
                track_full.extend([str(track_info) + '\n' for track_info in cfg.AUDIO.TRACK_PAD_BEFORE])
                track_full.extend(['{:.2f}'.format(track_info) + '\n' for track_info in track])
                track_full.extend([str(track_info) + '\n' for track_info in cfg.AUDIO.TRACK_PAD_AFTER])
            else:
                track_full.extend(['{:.2f}'.format(track_info) + '\n' for track_info in track])
            f_w.writelines(track_full)

            seq_length = tvs.shape[0]
            f_w.writelines([' '.join(['{:.5f}'.format(tvs_d) for tvs_d in tvs[idx, :]]) + '\n' for idx in range(seq_length)])

    def save_articulatory_info(self, tvs_pred, track_pred, trm_params_save_path):
        '''
        save extracted tvs and track for further use
        input: single sample of tvs, track
        HINT: the input tvs will be interpolated to have time interval of 4ms first, then the unnormalized  tvs and track will be saved
        '''
        # Here, original sample rate is controlled by time step of spec frame,
        # we expect sample rate to be trm input_control_rate 
        tvs = self.interpolate_tvs(tvs_pred, sample_rate=1. / (cfg.AUDIO.FRAME_SHIFT / 1000. * cfg.AUDIO.INPUT_CONTROL_RATE))
        tvs = self.unnormalize_tvs(tvs)
        track = self.unnormalize_track(track_pred)
        self.save_trm_params(tvs, track, trm_params_save_path, pad=True)
        return