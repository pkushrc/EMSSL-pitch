#coding=utf-8
import os
import sys
import time
import multiprocessing
import numpy as np
from tqdm import tqdm
import torch
import soundfile as sf
sys.path.append('..')
sys.path.append('../misc/')
from misc.config import cfg
from misc.preprocess_audio_trm import audioProcess
from misc.utils import get_filename, get_trm_filename, mkdir_p
from misc.dataset import AudioDataset, AudioDatasetFromRaw, audio_collate_fn # for audio
from misc.dataset import DataLoaderX, DataPrefetcher
from misc.dataset import preprocess_raw_data_multi_processes, preprocess_raw_data_single_batch
from misc.audio_from_trm import generate_audio_multi_processes
from train.model_pitch import Audio2Tvs

class A2TUtil(object):
    '''
    Load A2T model, and use the model to extract trm parameters from audio, save to files
    Accept either a2t model object or saved model file(file path specified in config) 
    '''
    def __init__(self,
                 moments_path,
                 model=None):
        """
        Init model parameters 
        """
        self.moments_path = moments_path
        self.model = model
        self.use_cuda = torch.cuda.is_available()
        self.batch_size = cfg.UTIL.BATCH_SIZE
        self.drop_r0 = cfg.AUDIO.TVS_DROP_R0 # if true, the extracted tvs will not have r0 dim
        self.audio_processor = audioProcess(moments_path)
        self.__build_and_init_model()

    def __build_and_init_model(self):
        if not self.model: # not receive a model as input
            self.model = Audio2Tvs()
            if self.use_cuda:
                self.model.cuda() # TODO: may support multi gpu
            
            model_path = os.path.join(cfg.UTIL.PRETRAINED_MODEL_DIR, cfg.UTIL.MODEL_FP)
            if len(model_path) > 0:
                print('Loading model parameters from {}'.format(model_path))
                if not self.use_cuda:
                    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                else:
                    checkpoint = torch.load(model_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print('Successfully restored model.')
            else:
                raise Exception('Please give model path to restore from !')
        else:
            pass # use input model

    def _extract_articulatory_info_base(self, audio_fps, processed_audio):
        '''
        Base method of extraction
        Return a list of tvs and track, [tvs1, tvs2, ...], [track1, track2, ...]
        Input:
            processed_audio: whether the audio fps have been processed
        Done: make sure order is conserved -- checked
        '''
        # set mode of model
        self.model.eval()

        # Prepare data
        self.dataset = AudioDataset(audio_fps) if processed_audio else AudioDatasetFromRaw(audio_fps, self.moments_path)
        self.data_loader = DataLoaderX(self.dataset, batch_size=self.batch_size, shuffle=False,
                                        num_workers=cfg.UTIL.NUM_LOADER_WORKERS, collate_fn=audio_collate_fn, pin_memory=True)
        if self.use_cuda:
            pbar = tqdm(enumerate(DataPrefetcher(self.data_loader)), total=len(self.data_loader))
        else:
            pbar = tqdm(enumerate(self.data_loader), total=len(self.data_loader))
        pbar.set_description("Extracting tvs")

        # Extract articulatory info
        tvs_pred_final = []
        track_pred_final = []
        spec_final = []
        with torch.no_grad():
            for _, (batch_spec, batch_seq_len) in pbar:
                tvs_pred, track_pred = self.model(batch_spec)

                batch_spec = batch_spec.cpu().numpy()
                tvs_pred = tvs_pred.cpu().numpy()
                tvs_pred = self.audio_processor.insert_r0(tvs_pred, self.drop_r0)
                track_pred = track_pred.cpu().numpy()
                

                num_sample = batch_spec.shape[0]
                tvs_pred_final.extend([tvs_pred[idx,:, :, :] for idx in range(num_sample)])
                track_pred_final.extend([track_pred[idx,:,:] for idx in range(num_sample)])
                spec_final.extend([batch_spec[idx, :, :, 0] for idx in range(num_sample)])
                    
        return tvs_pred_final, track_pred_final, spec_final
    
    def extract_and_save_articulatory_info(self, audio_fps, trm_params_fps_to_save, processed_audio=False, num_processes=4):
        '''
        Extract and save articulatory trajectories of audios
        Input audio_fps and trm_params_fps_to_save must corresponding to each other
        '''
        # Make sure inputs are list
        if not isinstance(audio_fps, list):
            assert(not isinstance(trm_params_fps_to_save, list))
            audio_fps = [audio_fps]
            trm_params_fps_to_save = [trm_params_fps_to_save]
        assert(len(audio_fps) == len(trm_params_fps_to_save))

        tvs_pred, track_pred, _ = self._extract_articulatory_info_base(audio_fps, processed_audio=processed_audio)
        if cfg.TRAIN.LOSS_LAMBDA == 0:
            print('Using given standard fake track...')
            normalized_track_fake = np.array([track_info for track_info in cfg.AUDIO.NORMALIZED_TRACK_FAKE])
            track_pred = [normalized_track_fake for i in range(len(track_pred))]

        print('Saving trm params...')
        start_ = time.time()
        pool = multiprocessing.Pool(num_processes)
        for tvs, track, trm_fp in zip(tvs_pred, track_pred, trm_params_fps_to_save):
            tvs1 = tvs[0]
            tvs2 = tvs[1]
            track1 = track[0]
            track2 = track[1]
            trm_fp1 = trm_fp
            trm_fp2 = trm_fp.replace('s1','s2')

            pool.apply_async(self.audio_processor.save_articulatory_info, (tvs1, track1, trm_fp1))
            pool.apply_async(self.audio_processor.save_articulatory_info, (tvs2, track2, trm_fp2))
            
        pool.close() 
        pool.join() # wait until all processes in pool are done, within the pool, there is no order
        end_ = time.time()
        print('{:4f}s used to save tvs'.format(end_ - start_))


def extract_synthesize_preprocess_from_ori_spec(A2TModel, gnuspeech_path, moments_path, ref_spec_fps, source_dir, target_dir):
    '''
    '''
    def get_middle_fps(ref_spec_fps, source_dir, target_dir):
        # Get fps for extracted trm parameters
        extracted_trm_params_fps = []
        for fp in ref_spec_fps:
            trm_fp_s1 = fp.replace(source_dir.strip('/'), target_dir.strip('/')).replace('_spec.pkl', '_trm_params_s1.txt')
            trm_fp_s2 = fp.replace(source_dir.strip('/'), target_dir.strip('/')).replace('_spec.pkl', '_trm_params_s2.txt')
            mkdir_p(trm_fp_s1.strip(get_filename(trm_fp_s1))) # recursively make dir
            mkdir_p(trm_fp_s2.strip(get_filename(trm_fp_s2))) # recursively make dir
            extracted_trm_params_fps.append(trm_fp_s1)
            
        # Get fps for synthetic audio fps
        synthetic_audio_fps = [fp.strip().replace('_trm_params_s1.txt', '_wav_output.wav') for fp in extracted_trm_params_fps]
        # Get fps for processed audios and trm parameters
        processed_spec_trm_fps = [fp.replace('_wav_output.wav', '_spec_tvs_track.pkl') for fp in synthetic_audio_fps]
        return extracted_trm_params_fps, synthetic_audio_fps, processed_spec_trm_fps
    
    # Generate middle fps
    extracted_trm_params_fps, synthetic_audio_fps,processed_spec_trm_fps = get_middle_fps(ref_spec_fps, source_dir, target_dir)
    # Extrct trm parameters and save
    A2TModel.extract_and_save_articulatory_info(ref_spec_fps, extracted_trm_params_fps, \
                                                processed_audio=True, num_processes=cfg.TRAIN.NUM_PROCESSES) #20 #32
   
    # Generate audio from tvs
    print('generating audio from trm_params...')
    start_ = time.time()
    generate_audio_multi_processes(extracted_trm_params_fps, synthetic_audio_fps, \
                                    gnuspeech_path, num_processes=cfg.TRAIN.NUM_PROCESSES) #24 #16
    end_ = time.time()
    print('{:3f}s used to generate audios from trm_params'.format(end_ - start_))
    # process reconstructed data
    start_ = time.time()
    # preprocess_raw_data_multi_processes(synthetic_audio_fps, extracted_trm_params_fps, \
    #        processed_spec_trm_fps, moments_path, num_processes=cfg.TRAIN.NUM_PROCESSES//3) # 8 single can exceed 100%
    preprocess_raw_data_single_batch(synthetic_audio_fps, extracted_trm_params_fps, \
                                     processed_spec_trm_fps, moments_path) # single can exceed 100%
    end_ = time.time()
    print('{:3f}s used to process newly generated audio and trm_params'.format(end_ - start_))

    return processed_spec_trm_fps