# coding=utf-8
import os
import sys
import numpy as np
from datetime import datetime
import time
from tqdm import tqdm
from functools import reduce
import glob
import random
import pickle
import argparse

import torch
from torch import nn, optim
from torch.utils import tensorboard

sys.path.append('..')
sys.path.append('../misc/')
sys.path.append('../model_util')
from misc.config import cfg
from misc.utils import mkdir_p, get_filename
from misc.plot_func import plot_func
from misc.preprocess_audio_trm import audioProcess
from misc.dataset import A2TDataset, collate_fn
from misc.dataset import DataLoaderX, DataPrefetcher
from model_pitch import Audio2Tvs, MseLoss
from model_util.a2t_model_util import A2TUtil, extract_synthesize_preprocess_from_ori_spec

def setup_seed(seed):
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True   # if benchmark=True, deterministic will be False
        torch.backends.cudnn.deterministic = False
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed)
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=16, linewidth=120) #profile='full'

WORKER_SEED = 1
def worker_init(worker_id):
    np.random.seed(int(WORKER_SEED + worker_id))

class AverageMeter(object):
    def __init__(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
  
class A2TTrainer(object):
    '''
    Train and test Audio2Tvs model
    '''
    def __init__(self,
                 moments_path, 
                 test_model_dir=None,
                 test_model_fp=None,
                 ckt_logs_dir='ckt_logs',
                 exp_name='a2t_model_'+datetime.now().strftime('%Y%m%d-%H:%M:%S')):
        self.model = None
        self.model_path = ''
        self.restore_model = not cfg.TRAIN.FLAG or cfg.TRAIN.RESTORE
        self.adapt = cfg.TRAIN.ADAPT # if adapt = True, will not summary audio and spec
        self.optimizer = None
        self.lr_scheduler = None
        self.epoch = 0
        self.global_step = 0

        self.plot_func = plot_func()
        self.audio_processor = audioProcess(moments_path)
        self.moments_path = moments_path
        self.drop_r0 = cfg.AUDIO.TVS_DROP_R0
        self.use_cuda = torch.cuda.is_available()
        self.N_GPU = len(cfg.TRAIN.GPU_ID)
        self.exp_name = exp_name

        if not cfg.TRAIN.FLAG: # test
            ckt_logs_dir = test_model_dir if test_model_dir else cfg.UTIL.PRETRAINED_MODEL_DIR
            ckt_model_fp = test_model_fp if test_model_fp else cfg.UTIL.MODEL_FP
            self.model_path = os.path.join(ckt_logs_dir, ckt_model_fp)
        elif cfg.TRAIN.RESTORE: # train from restored model
            self.model_path = os.path.join(cfg.TRAIN.PRETRAINED_MODEL_DIR, cfg.TRAIN.MODEL_FP)
            if cfg.TRAIN.INPLACE:
                ckt_logs_dir = cfg.TRAIN.PRETRAINED_MODEL_DIR
            else:
                ckt_logs_dir = os.path.join(ckt_logs_dir, self.exp_name)
                mkdir_p(ckt_logs_dir)
                print('Restored model but save log to {}'.format(ckt_logs_dir))
        else:
            ckt_logs_dir = os.path.join(ckt_logs_dir, self.exp_name)
            mkdir_p(ckt_logs_dir)
        self.log_dir = ckt_logs_dir
        self.iter_train_log_dir = os.path.join(self.log_dir, 'iter_train')
        
        self.LR_STARTER = cfg.TRAIN.LR
        self.LR_DECAY_EPOCH = cfg.TRAIN.LR_DECAY_EPOCH
        self.LR_DECAY_RATE = cfg.TRAIN.LR_DECAY_RATE

        self.writer = tensorboard.SummaryWriter(self.log_dir)
        self.__build_model()

    def __print_model_info(self, model):
        nparams = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                p_shape = list(param.shape)
                p_n = reduce(lambda x, y: x * y, p_shape)
                nparams += p_n
                print('{} ({}): {}'.format(p_shape, p_n, name))
        print('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))
        print('-' * 80)

    def __add_figure_summary(self, batch_spec, batch_tvs_gt, batch_tvs_pred, \
                                batch_track_gt, batch_track_pred, batch_seq_len, label):
        num_samples = batch_spec.shape[0]
        for i in range(num_samples):
            self.writer.add_figure('{}/spec/sample{}'.format(label, i), 
                                    self.plot_func.plot_spec([batch_spec[i, :batch_seq_len[i], :]]), 
                                    self.global_step)
            self.writer.add_figure('{}/tvs/sample0{}'.format(label, i),
                                    self.plot_func.plot_tvs([self.audio_processor.insert_r0(batch_tvs_gt[i,0, :batch_seq_len[i], :], self.drop_r0), 
                                                             self.audio_processor.insert_r0(batch_tvs_pred[i,0, :batch_seq_len[i], :], self.drop_r0)]), 
                                    self.global_step)
            self.writer.add_figure('{}/tvs/sample1{}'.format(label, i),
                                    self.plot_func.plot_tvs([self.audio_processor.insert_r0(batch_tvs_gt[i,1, :batch_seq_len[i], :], self.drop_r0), 
                                                             self.audio_processor.insert_r0(batch_tvs_pred[i,1, :batch_seq_len[i], :], self.drop_r0)]), 
                                    self.global_step)
            self.writer.add_figure('{}/track/sample0{}'.format(label, i),
                                    self.plot_func.plot_and_compare_track([batch_track_gt[i,0, :], batch_track_pred[i,0, :]]),
                                    self.global_step)
            self.writer.add_figure('{}/track/sample0{}'.format(label, i),
                                    self.plot_func.plot_and_compare_track([batch_track_gt[i,1, :], batch_track_pred[i,1, :]]),
                                    self.global_step)
    
    def __add_audio_summary(self, batch_wav, batch_wav_len, batch_sr, label):
        num_samples = batch_wav.shape[0]
        for i in range(num_samples):
            self.writer.add_audio('{}/sample{}'.format(label, i),
                                    batch_wav[i, :batch_wav_len[i]], 
                                    self.global_step,
                                    sample_rate=int(batch_sr[i]))

    def __save_checkpoint(self, epoch, cur_result):
        checkpoint = {}

        data_p = isinstance(self.model, nn.DataParallel)
        checkpoint['model_state_dict'] = self.model.module.state_dict() if data_p else self.model.state_dict()
        checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['global_step'] = self.global_step

        checkpoint_path = os.path.join(self.log_dir, "MODEL_EPOCH_{}_loss={:.4f}.pkl".format(epoch, cur_result))
        torch.save(checkpoint, checkpoint_path)
        print('Saved checkpoint into {} !'.format(checkpoint_path))
    
    def __save_train_fps(self, epoch, train_fps):
        '''
        Since there is random process in choosing training data, save train fps for restoring
        '''
        train_fps_path = os.path.join(self.log_dir, "TRAIN_FPS_EPOCH_{}.pkl".format(epoch))
        torch.save(train_fps, train_fps_path)
        print('Saved train fps into {} !'.format(train_fps_path))

    def __save_state(self, state_fp):
        '''
        Just for debugging
        '''
        checkpoint = {}

        data_p = isinstance(self.model, nn.DataParallel)
        checkpoint['model_state_dict'] = self.model.module.state_dict() if data_p else self.model.state_dict()
        checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, state_fp)
        print('Saved states into {} !'.format(state_fp))

    def __init_parameters(self, restore=False):
        """
        Initial parameters and part of hyper-parameters for model
        :param restore: checkpoint
        :return: None
        :TODO: seems there is a problem with restore of learning rate
        """
        if restore:
            if len(self.model_path) > 0:
                print('Loading model parameters from {}'.format(self.model_path))

                checkpoint = torch.load(self.model_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.epoch = checkpoint['epoch'] + 1
                self.global_step = checkpoint['global_step'] + 1
                print('Successfully restored model.')
            else:
                raise Exception('Please give model path to restore from !')
        else:
            print("Created model with fresh parameters.")
            # parameters have been initalized in model building
            self.epoch = 0
            self.global_step = 0

    def __build_model(self):
        """
        Create a model and its optimizer, scheduler, etc.
        :return: None
        """
        print('Building model...')
        self.model = Audio2Tvs()
        self.__print_model_info(self.model)
        if self.use_cuda:
            print('{} GPUs specified to use, {} GPUs available.'.format(self.N_GPU, torch.cuda.device_count()))
            print('torch version: {}, cuda version: {}, cudnn version: {}'.format(torch.__version__, torch.version.cuda, torch.backends.cudnn.version()))
            self.model.cuda()

        self.loss = MseLoss()
        

        self.optimizer = optim.Adam(self.model.parameters(), self.LR_STARTER, betas=(0.5, 0.999))
        # self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.LR_DECAY_RATE) # 600 steps to 0.5

        self.__init_parameters(restore=self.restore_model) # should before dataparallel to make dict names match
        
        if self.use_cuda and self.N_GPU > 1:
            gpu_ids = [i for i in range(self.N_GPU)]
            self.model = nn.DataParallel(self.model, device_ids=gpu_ids)

        print('Succesfully built model!')

    def _train(self, cur_epoch, img_summary, init_img_compare):
        """
        Train the model once, i.e., one epoch
        :return:
        """
        self.model.train()

        log_interval = cfg.TRAIN.LOG_INTERVAL

        tvs_loss_mse_meter = AverageMeter()
        track_loss_meter = AverageMeter()

        if self.use_cuda:
            pbar = tqdm(enumerate(DataPrefetcher(self.train_dataloader)), total=len(self.train_dataloader))
        else:
            pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))        
        start_time = time.time()
        for i, (batch_wav, batch_wav_len, batch_sr, batch_spec, batch_tvs, batch_track, batch_seq_len) in pbar:
        # for i, (batch_spec, batch_tvs, batch_track, batch_seq_len) in pbar:
            prepare_time = start_time - time.time()
            
            # Clean up gram & Forward Prop & Compute Loss
            self.optimizer.zero_grad()
            tvs_pred, track_pred = self.model(batch_spec)
            forward_time = start_time - time.time() - prepare_time
        
            # Compute Loss
            cur_loss_tvs_mse, cur_loss_track = self.loss(batch_tvs, tvs_pred, batch_track, track_pred)
            cur_loss = cur_loss_tvs_mse + cfg.TRAIN.LOSS_LAMBDA * cur_loss_track

            # Backward Prop.
            cur_loss.backward()

            # Update Weights
            self.optimizer.step()
            backward_time = start_time - time.time() - prepare_time - forward_time

            # Keep track of efficiency
            process_time = start_time - time.time() - prepare_time
            pbar.set_description("Train efficiency: {:.2f}, fw: {:.2f}, bw: {:.2f}, epoch:{}".format(
                process_time / (process_time + prepare_time), forward_time / (process_time + prepare_time), \
                backward_time / (process_time + prepare_time), cur_epoch))

            # Add summary to tensorboard
            tvs_loss_mse_meter.update(cur_loss_tvs_mse.detach().cpu().numpy())
            track_loss_meter.update(cur_loss_track.detach().cpu().numpy())
            if (i + 1) % log_interval == 0 or i == len(self.train_dataloader) - 1:
                self.writer.add_scalar('train/tvs_loss_mse', cur_loss_tvs_mse, self.global_step)
                self.writer.add_scalar('train/track_loss', cur_loss_track, self.global_step)
                self.writer.add_scalar('train/object_loss', cur_loss, self.global_step)
            
            # if i == len(self.train_dataloader) - 1:
            #     print('train_tvs_loss_mse: {}, train_track_loss: {} eval at epoch: {}'.format( \
            #         tvs_loss_mse_meter.avg, track_loss_meter.avg, cur_epoch))
            if (img_summary and i == len(self.train_dataloader) - 1) or (init_img_compare and i == 0): # the first batch to compare tvs
                sample_num = min(cfg.EVAL.SAMPLE_NUM, batch_spec.shape[0])
                self.__add_figure_summary(batch_spec[:sample_num, :, :, 0].detach().cpu().numpy(),
                                          batch_tvs[:sample_num,:, :, :].detach().cpu().numpy(),
                                          tvs_pred[:sample_num,:, :, :].detach().cpu().numpy(),
                                          batch_track[:sample_num,:, :].detach().cpu().numpy(),
                                          track_pred[:sample_num,:, :].detach().cpu().numpy(),
                                          batch_seq_len[:sample_num].detach().cpu().numpy(),
                                          label='train')
            if img_summary and i == len(self.train_dataloader) - 1: # the last batch may not have enough samples
                sample_num = min(cfg.EVAL.SAMPLE_NUM, batch_spec.shape[0])
                self.__add_audio_summary(batch_wav[:sample_num, :].detach().cpu(),
                                          batch_wav_len[:sample_num].detach().cpu(),
                                          batch_sr[:sample_num].detach().cpu().numpy(),
                                          label='train')

            # update global step and step start time
            self.global_step += 1
            start_time = time.time()
        return tvs_loss_mse_meter.avg, track_loss_meter.avg

    def _eval(self, data_loader, task='eval'):
        """
        Evaluate the model once, i.e., one epoch
        :return:
        """
        self.model.eval()

        tvs_loss_mse_meter = AverageMeter()
        track_loss_meter = AverageMeter()

        if self.use_cuda:
            pbar = tqdm(enumerate(DataPrefetcher(data_loader)), total=len(data_loader))
        else:
            pbar = tqdm(enumerate(data_loader), total=len(data_loader))
        pbar.set_description("Evaluating")
        with torch.no_grad():
            for i, (batch_wav, batch_wav_len, batch_sr, batch_spec, batch_tvs, batch_track, batch_seq_len) in pbar:
            # for i, (batch_spec, batch_tvs, batch_track, batch_seq_len) in pbar:
                # Forward Prop & Compute Loss
                #print(batch_wav.shape, batch_wav_len.shape, batch_sr.shape, batch_spec.shape, batch_tvs.shape, batch_track.shape, batch_seq_len.shape)
                tvs_pred, track_pred = self.model(batch_spec)
                #print(tvs_pred.shape,track_pred.shape)
                # Compute Loss
                cur_loss_tvs_mse, cur_loss_track  = self.loss(batch_tvs, tvs_pred, batch_track, track_pred)
                cur_loss = cur_loss_tvs_mse + cfg.TRAIN.LOSS_LAMBDA * cur_loss_track

                # Log
                tvs_loss_mse_meter.update(cur_loss_tvs_mse.detach().cpu().numpy())
                track_loss_meter.update(cur_loss_track.detach().cpu().numpy())

                if not self.adapt and task == 'eval' and i == len(data_loader) - 1: # just use first few samples
                    sample_num = min(cfg.EVAL.SAMPLE_NUM, batch_spec.shape[0])
                    self.__add_figure_summary(batch_spec[:sample_num, :, :, 0].detach().cpu().numpy(),
                                              batch_tvs[:sample_num,:, :, :].detach().cpu().numpy(),
                                              tvs_pred[:sample_num, :,:, :].detach().cpu().numpy(),
                                              batch_track[:sample_num,:, :].detach().cpu().numpy(),
                                              track_pred[:sample_num,:, :].detach().cpu().numpy(),
                                              batch_seq_len[:sample_num].detach().cpu().numpy(),
                                              label=task)
                    self.__add_audio_summary(batch_wav[:sample_num, :].detach().cpu(),
                                             batch_wav_len[:sample_num].detach().cpu(),
                                             batch_sr[:sample_num].detach().cpu().numpy(),
                                             label=task)
                
        if task == 'eval':
            self.writer.add_scalar('{}/tvs_loss_mse'.format(task), tvs_loss_mse_meter.avg, self.global_step)
            self.writer.add_scalar('{}/track_loss'.format(task), track_loss_meter.avg, self.global_step)
            cur_object_loss = tvs_loss_mse_meter.avg + cfg.TRAIN.LOSS_LAMBDA * track_loss_meter.avg
            self.writer.add_scalar('{}/object_loss'.format(task), cur_object_loss, self.global_step)
            self.writer.add_scalar('learning rate', self.optimizer.param_groups[0]['lr'], self.global_step)

        torch.cuda.empty_cache()

        return tvs_loss_mse_meter.avg, track_loss_meter.avg

    def train(self, train_data_fps, eval_data_fps, mode=None):
        # init data
        self.train_dataset = A2TDataset(train_data_fps, drop_r0=self.drop_r0)
        self.eval_dataset = A2TDataset(eval_data_fps, drop_r0=self.drop_r0)
        self.train_dataloader = DataLoaderX(self.train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                           num_workers=cfg.TRAIN.NUM_LOADER_WORKERS, collate_fn=collate_fn, 
                                           pin_memory=True, worker_init_fn=worker_init)
        self.eval_dataloader = DataLoaderX(self.eval_dataset, batch_size=cfg.EVAL.BATCH_SIZE, shuffle=False,
                                           num_workers=cfg.EVAL.NUM_LOADER_WORKERS, collate_fn=collate_fn, 
                                           pin_memory=True, worker_init_fn=worker_init, drop_last=False)
        
        INIT_EPOCH = self.epoch
        print('Total number of train data: {}, eval data: {}'.format(len(self.train_dataset), len(self.eval_dataset)))
        print('Start at epoch: {}, step: {}'.format(self.epoch, self.global_step))
        # self.writer.add_graph(self.model, input_to_model=torch.Tensor(16, 160, 80 ,1).zero_(), verbose=False) # TODO: relocate it

        end_epoch = cfg.ITERTRAIN.EPOCH_PER_ITERATE + INIT_EPOCH if mode == 'iterate' else cfg.TRAIN.MAX_EPOCH
        for epoch in range(INIT_EPOCH, end_epoch):
            # if need evaluation or need to save to checkpoint, then do evaluation
            do_evaluation = (epoch + 1) % cfg.EVAL.SAMPLE_PER_EPOCH == 0 or \
                            (epoch + 1) % cfg.TRAIN.SAMPLE_PER_EPOCH == 0 or \
                            (epoch + 1) % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or \
                            (epoch + 1) == end_epoch
            init_img_compare = epoch == INIT_EPOCH

            # train model
            if not self.adapt:
                eval_tvs_loss_mse, eval_track_loss = self._train(epoch, img_summary=do_evaluation, init_img_compare=init_img_compare)
            else:
                eval_tvs_loss_mse, eval_track_loss = self._train(epoch, img_summary=False, init_img_compare=False)

            # do validation
            if do_evaluation:
                eval_tvs_loss_mse, eval_track_loss = self._eval(self.eval_dataloader, task='eval')
                print('eval_tvs_loss_mse: {}, eval_track_loss: {} eval at epoch: {}'.format( \
                    eval_tvs_loss_mse, eval_track_loss, epoch))

            # save to checkpoint
            if (epoch + 1) % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or (epoch + 1) == end_epoch:
                cur_result = eval_tvs_loss_mse + cfg.TRAIN.LOSS_LAMBDA * eval_track_loss
                self.__save_checkpoint(epoch, cur_result)
                if cfg.TRAIN.OLD_SAMPLE_DROP_RATE != 1:
                    self.__save_train_fps(epoch, train_data_fps)
            
            # TODO: may add early stop
            
            # schedule learning rate, the step counter is not global step but the times the scheculer is called
            # if (epoch + 1) % cfg.EVAL.SAMPLE_PER_EPOCH == 0:
                # self.lr_scheduler.step(eval_tvs_loss + eval_track_loss)
            if (epoch + 1) % self.LR_DECAY_EPOCH == 0:
                self.lr_scheduler.step() # for exponentially decay
            self.epoch += 1
            
            torch.cuda.empty_cache()
            
    def eval_loss(self, test_data_fps):
        '''
            Just calculate loss, to evaluate model performance, refer to ../model_util/a2t_model_test.py
        '''
        # init model
        assert(cfg.TRAIN.FLAG == False and cfg.TRAIN.RESTORE == True)
        # self.__init_parameters(restore=True) # have restored in self.__init__
        # init data
        self.test_dataset = A2TDataset(test_data_fps, drop_r0=self.drop_r0)
        self.test_dataloader = DataLoaderX(self.test_dataset, batch_size=cfg.UTIL.BATCH_SIZE, shuffle=True,
                                           num_workers=cfg.UTIL.NUM_LOADER_WORKERS, collate_fn=collate_fn, 
                                           pin_memory=True, worker_init_fn=worker_init, drop_last=False)
        print('Start testing {} fps...'.format(len(test_data_fps)))
        test_tvs_loss_mse, test_track_loss = self._eval(self.test_dataloader, task='test')
        test_object_loss = test_tvs_loss_mse + cfg.TRAIN.LOSS_LAMBDA * test_track_loss
        print('Overall tvs mseloss: {:4f}, track loss: {:4f}'.format(test_tvs_loss_mse, test_track_loss))

        return test_object_loss, test_tvs_loss_mse, test_track_loss        

    def iterative_train(self, train_spec_fps, eval_spec_fps, raw_audio_dir, gnuspeech_path):
        '''
        Input:
            train_spec_fps: spectrograms of training audios
            eval_spec_fps: spectrograms of evaluation audios
            raw_audio_dir: dir containing train_spec_fps and eval_spec_fps, should have 'train', 'eval' splits
            gnuspeech_path: dir contains gnuspeech app
        '''
        # Load historical training data fps when restore model
        if cfg.TRAIN.FLAG and cfg.TRAIN.RESTORE and cfg.TRAIN.OLD_SAMPLE_DROP_RATE != 1 and cfg.TRAIN.START_FPS:
            train_start_fps = os.path.join(cfg.TRAIN.PRETRAINED_MODEL_DIR, cfg.TRAIN.START_FPS)
            processed_train_fps = torch.load(train_start_fps)
            print('Loaded {} train fps from: {}.'.format(len(processed_train_fps), train_start_fps))
        else:
            processed_train_fps = []
        
        for i in range(cfg.ITERTRAIN.ITERATE_START_N, cfg.ITERTRAIN.ITERATE_N):
            # drop some historical training samples
            random.shuffle(processed_train_fps)
            processed_train_fps = processed_train_fps[:int(len(processed_train_fps) * (1 - cfg.TRAIN.OLD_SAMPLE_DROP_RATE))]
            all_size = (len(processed_train_fps) + len(train_spec_fps))
            last_size = all_size % cfg.TRAIN.BATCH_SIZE
            if last_size // self.N_GPU < 2:
                processed_train_fps = processed_train_fps[:(len(processed_train_fps) - last_size)]
    
            # Sample new training data
            data_save_dir = os.path.join(self.iter_train_log_dir, 'iter_{}'.format(i + 1))
            A2TModel = A2TUtil(self.moments_path, model=self.model)
            print('Preparing new training data...')
            new_processed_train_fps = extract_synthesize_preprocess_from_ori_spec(A2TModel, \
                                        gnuspeech_path, self.moments_path, train_spec_fps, raw_audio_dir, data_save_dir)
            processed_train_fps.extend(new_processed_train_fps) # shuffle will be done during training

            # prepare validation data
            if self.adapt:
                processed_eval_fps = new_processed_train_fps
            else:
                print('Preparing new evaluation data...')
                processed_eval_fps = extract_synthesize_preprocess_from_ori_spec(A2TModel, \
                                     gnuspeech_path, self.moments_path, eval_spec_fps, raw_audio_dir, data_save_dir)

            # Train model with updated data
            # During every iteration, lr will be reset
            self.optimizer = optim.Adam(self.model.parameters(), self.LR_STARTER, betas=(0.5, 0.999))
            # self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3)
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.LR_DECAY_RATE) # 600 steps to 0.5
            self.train(processed_train_fps, processed_eval_fps, mode='iterate') 


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(','.join([str(idx) for idx in cfg.TRAIN.GPU_ID]))
    setup_seed(666)
    exp_name = 'a2t_model_{}_'.format(cfg.TRAIN.EXP_NAME)+ datetime.now().strftime('%Y%m%d-%H:%M:%S')
    a2t_trainer = A2TTrainer(cfg.GLOBAL.MOMENTS_PATH, ckt_logs_dir=cfg.TRAIN.LOG_DIR, exp_name=exp_name)
    print('Using moments from: {}'.format(cfg.GLOBAL.MOMENTS_PATH))
    
    processed_audio_dir = cfg.TRAIN.PROCESSED_AUDIO_DIR
    print('Audios from: {}'.format(processed_audio_dir))
    train_spec_fps = glob.glob(os.path.join(processed_audio_dir, 'train/*_spec.pkl')) # should have been preprocessed
    eval_spec_fps = glob.glob(os.path.join(processed_audio_dir, 'eval/*_spec.pkl'))
    print('Num of audios to fit: {}, to eval: {}'.format(len(train_spec_fps), len(eval_spec_fps)))
    a2t_trainer.iterative_train(train_spec_fps, eval_spec_fps, processed_audio_dir, cfg.GLOBAL.GNUSPEECH_DIR)
