#coding=utf-8
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from pylab import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn import preprocessing

from preprocess_audio_trm import audioProcess
from config import cfg
from utils import get_filename

class plot_func:
    def __init__(self):
        self.audio_processor = audioProcess()
        self.tvs_label = cfg.AUDIO.TVS_DIM_NAME

    ####################################################################
    # Following are sepc related plot methods
    def plot_signal_t(self, audio_fp):
        '''
        plot audio signal in time domain
        input: 
            audio file path
        '''
        _, signal = self.audio_processor.load_wav(audio_fp, sr=cfg.AUDIO.FS)
        time = 1.0 * np.array(range(0, len(signal))) / cfg.AUDIO.FS
        plt.plot(time, signal, 'b-', linewidth=1)
        plt.ylabel('Amplitude')
        plt.xlabel('Time/s')
        plt.title('audio signal of {}'.format(get_filename(audio_fp)))
        plt.show()
        # seq_len = len(signal)
        # print('total length: {}'.format(seq_len))
        # start_time = 1.66
        # end_time = 1.785
        # start_i = int(start_time * cfg.AUDIO.FS)
        # end_i = int(end_time * cfg.AUDIO.FS)
        # plt.plot(time[start_i : end_i], signal[start_i : end_i], 'b-', linewidth=1)
        # start_i = int(start_time * cfg.AUDIO.FS)//1000
        # for i in range(start_i, seq_len // 1000):
        #     plt.plot(time[i*1000 : (i + 1)*1000], signal[i*1000 : (i + 1)*1000], 'b-', linewidth=1)
        #     plt.ylabel('Amplitude')
        #     plt.xlabel('Time/s')
        #     plt.title('first 1000 point of signal')
        #     plt.show()
        

    def plot_and_compare_signal_t(self, signal1, signal2):
        '''
        plot original and reconstructed signal in time domain
        input: time series of audio signal
        '''
        # if len(signal1)!=len(signal2):
        #     print("error in plot_signal1")

        time1 = 1.0 * np.array(range(0, len(signal1))) / cfg.AUDIO.FS
        time2 = 1.0 * np.array(range(0, len(signal2))) / cfg.AUDIO.FS

        plt.subplot(311)
        # plt.plot(time1, preprocessing.maxabs_scale(signal1), 'b-', linewidth=1, label='original')
        plt.plot(time1, signal1, 'b-', linewidth=1, label='original')
        plt.legend(loc='upper right')
        plt.ylabel('Amplitude')

        plt.subplot(312)
        # plt.plot(time2, preprocessing.maxabs_scale(signal2), 'r-', linewidth=1, label='reconstructed')
        plt.plot(time2, signal2, 'r-', linewidth=1, label='reconstructed')
        plt.legend(loc='upper right')
        plt.ylabel('Amplitude(dB)')

        plt.subplot(313)
        # plt.plot(time2, preprocessing.maxabs_scale(signal2), 'r-', linewidth=1, label='reconstructed')
        assert(len(signal1) == len(signal2))
        plt.plot(time1, signal1 - signal2, 'y-', linewidth=1, label='deviation')
        plt.legend(loc='upper right')
        plt.ylabel('Amplitude(dB)')

        plt.xlabel('Time/s')
        plt.show()

    def plot_and_compare_signal_f(self, signal1, signal2, sr=cfg.AUDIO.FS):
        '''
        plot original and reconstructed signal in frequency domain
        input: time series of audio signal
        '''
        def fft_util(signal):
            signal_f = np.absolute(np.fft.fft(signal))
            signal_f[signal_f < 1e-30] = 1e-30
            return 10 * np.log10(signal_f)

        spectrum_1 = fft_util(signal1)
        spectrum_2 = fft_util(signal2)

        N=int(math.floor(len(spectrum_1)/2))+1
        x=np.linspace(0, sr/2, N)
        plt.plot(x, spectrum_1[0:N], 'b-', linewidth=1, label='original')
        plt.plot(x, spectrum_2[0:N], 'r-', linewidth=1, label='recontructed')
        plt.legend(loc='upper right')
        plt.xlabel('Frequency(Hz)')
        plt.ylabel('Amplitude(dB)')
        plt.show()
        

    def plot_spec_single_local(self, power_spec):
        num_freq_point=np.shape(power_spec)[1]
        plt.imshow(power_spec.transpose()[::-1], cmap=cm.seismic)
        plt.xlabel('frame index')
        plt.ylabel('Channel(mel)')
        # plt.yticks(np.linspace(0, power_spec.shape[-1], 3), ())
        plt.yticks((0, num_freq_point/2, num_freq_point-1), (num_freq_point, num_freq_point/2, 0))
        plt.colorbar(shrink=0.5, aspect=5)
        plt.show()

    def plot_spec_list_local(self, spec_list, col=2):
        spec_num = len(spec_list)
        # print("printing {} spectrums...".format(spec_num))
        row = spec_num // col
        fig = plt.figure(figsize=(10 * col, 5 * row))
        num_fbank = spec_list[0].shape[-1]
        for spec_idx in range(spec_num):
            ax = fig.add_subplot(row, col, spec_idx + 1)
            im = ax.imshow(spec_list[spec_idx].transpose()[::-1], cmap=cm.viridis)
            ax.set_xlabel('Frame')
            ax.set_ylabel('Mel bin')
            ax.set_yticks(np.linspace(0, num_fbank, 3))
            ax.set_yticklabels([num_fbank, num_fbank/2, 0])
            # fig.colorbar(im, ax=ax)
        plt.show()

    def plot_spec_list_local_save(self, spec_list, save_fp, col=1):
        spec_num = len(spec_list)
        # assert(spec_num == 2)
        # spec_len = min([spec.shape[0] for spec in spec_list])
        # spec_list = [spec[:spec_len, :] for spec in spec_list]
        idx_dict = dict([(0, 1), (1, 4), (2, 2), (3, 5), (4, 3), (5, 6)])
        spec_list_truncate = []
        for i in range(col):
            spec_len = min([spec_list[2 * i].shape[0], spec_list[2 * i + 1].shape[0]])
            spec_list_truncate.append(spec_list[2 * i][:spec_len, :])
            spec_list_truncate.append(spec_list[2 * i + 1][:spec_len, :])
        # print("printing {} spectrums...".format(spec_num))
        row = spec_num // col
        fig = plt.figure(figsize=(20 * col, 5 * row))
        for spec_idx in range(spec_num):
            ax = fig.add_subplot(row, col, idx_dict[spec_idx])
            im = ax.imshow(spec_list_truncate[spec_idx].transpose()[::-1], cmap=cm.viridis)
            ax.set_xlabel('Frame', fontsize=20)
            ax.set_ylabel('Mel Filterbank', fontsize=20)
            ax.set_yticks([0, 19, 39, 59, 79])
            ax.set_yticklabels([80, 60, 40, 20, 0])
            ax.tick_params(axis='y', labelsize=20)
            ax.tick_params(axis='x', labelsize=20)
            # fig.colorbar(im, ax=ax)
        plt.savefig(save_fp, dpi=600)
        plt.show()

    def plot_spec_list_local_single_save(self, spec_list, save_fp, col=1):
        spec_num = len(spec_list)
        # assert(spec_num == 2)
        spec_len = min([spec.shape[0] for spec in spec_list])
        spec_list = [spec[:spec_len, :] for spec in spec_list]
        # print("printing {} spectrums...".format(spec_num))
        row = spec_num // col
        for spec_idx in range(spec_num):
            fig = plt.figure(figsize=(40, 5))
            ax = fig.add_subplot(1, 1, 1)
            im = ax.imshow(spec_list[spec_idx].transpose()[::-1], cmap=cm.viridis)
            ax.set_xlabel('Frame', fontsize=20)
            ax.set_ylabel('Mel Filterbank', fontsize=20)
            ax.set_yticks([0, 19, 39, 59, 79])
            ax.set_yticklabels([80, 60, 40, 20, 0])
            ax.tick_params(axis='y', labelsize=20)
            ax.tick_params(axis='x', labelsize=20)
            # fig.colorbar(im, ax=ax)
            plt.savefig(save_fp.format(spec_idx), dpi=600)
            plt.show()
    
    def plot_spec(self, spec_list):
        fig = matplotlib.figure.Figure(figsize=(10, 6))
        spec_num = len(spec_list) if isinstance(spec_list, list) else spec_list.shape[0]
        num_fbank = spec_list[0].shape[-1]
        for spec_idx in range(spec_num):
            ax = fig.add_subplot(spec_num, 1, spec_idx + 1)
            im = ax.imshow(spec_list[spec_idx].transpose()[::-1], cmap=cm.seismic)
            ax.set_xlabel('Frame index')
            ax.set_ylabel('Channel(mel)')
            ax.set_yticks(np.linspace(0, num_fbank, 3))
            ax.set_yticklabels([num_fbank, num_fbank/2, 0])
            fig.colorbar(im, ax=ax)
        return fig

    def plot_and_save_spec(self, spec_ori, spec_pred, sequence_lengths, save_num, save_dir):
        for i in range(save_num):
            _spec_ori = spec_ori[i, :sequence_lengths[i], :, 0]
            _spec_pred = spec_pred[i, :sequence_lengths[i], :, 0]

            high_freq=cfg.AUDIO.HIGH_F
            spec_list = [_spec_ori, _spec_pred]
            spec_num = len(spec_list)
            fig = plt.figure(figsize=(10, 6))
            for spec_idx in range(spec_num):
                plt.subplot(spec_num, 1, spec_idx + 1)
                plt.imshow(spec_list[spec_idx].transpose()[::-1], cmap=cm.seismic)
                plt.xlabel('Frame index')
                plt.ylabel('Channel(mel)')
                plt.yticks(np.linspace(0, spec_list[spec_idx].shape[-1], 3), [high_freq, high_freq/2, 0])
                plt.colorbar()
            figname = os.path.join(save_dir, 't2a_spec_test_{}.png'.format(i))
            print('Save to {}...'.format(figname))
            plt.savefig(figname)
            plt.close()

    ####################################################################
    # Following are tvs, track related plot methods

    def plot_tvs_no_overlap_local(self, tvs_list, color='blue', restrict=False):
        '''
        input: a list of tvs, [tvs1, tvs2, ...],
        if restrict, y-axis will be restricted to [-1, 1]
        '''
        tvs_num = len(tvs_list)
        fig = plt.figure(figsize=(12 * tvs_num, 8))

        # plt.title('Original articulatory trajectories', size=12) 
        for tvs_idx in range(tvs_num):
            for tvs_dim in range(cfg.AUDIO.TVS_DIM):
                ax = fig.add_subplot(cfg.AUDIO.TVS_DIM, tvs_num, tvs_num * tvs_dim + tvs_idx + 1)
                ax.plot(range(0, tvs_list[tvs_idx].shape[0], 1), tvs_list[tvs_idx][:, tvs_dim], color=color)
                if tvs_idx == 0:
                    ax.set_ylabel(self.tvs_label[tvs_dim], rotation=0, verticalalignment='center', size=10)
                # ax.set_yticks([])
                # ax.set_yticks([np.max(tvs_list[tvs_idx, :, tvs_dim]), np.min(tvs_list[tvs_idx, :, tvs_dim])])
                if restrict: ax.set_yticks([-1 , 1])
                # if tvs_dim == cfg.AUDIO.TVS_DIM - 1:
                #     ax.set_xlabel('frame index')
        plt.xlabel('frame index', size=12)
        plt.show()
    
    def plot_and_compare_tvs_local(self, tvs_list):
        assert(len(tvs_list) == 2)
        tvs_num = 2
        sub_num = 3
        color_list  = ['green', 'red']
        for i in range(1):
            fig = plt.figure(figsize=(15, 10))
            for tvs_idx in range(tvs_num):
                for tvs_dim in range(cfg.AUDIO.TVS_DIM):
                    plt.subplot(cfg.AUDIO.TVS_DIM, sub_num, sub_num * tvs_dim + tvs_idx + 1)
                    plt.plot(range(len(tvs_list[tvs_idx])), tvs_list[tvs_idx][:, tvs_dim], color=color_list[tvs_idx])
                    tick_params(direction='in')
                    # plt.ylim(-1, 1)
                    # plt.yticks([-1, 1])
                    if tvs_idx == 0:
                        plt.ylabel(self.tvs_label[tvs_dim], rotation=0, verticalalignment='center')
                    if tvs_dim + 1 == cfg.AUDIO.TVS_DIM:
                        plt.xlabel('Frame index')
                    if tvs_dim + 1 != cfg.AUDIO.TVS_DIM:
                        plt.xticks([])
                    # ax.set_yticks([]) 
                    # ax.set_yticks([np.max(tvs_list[tvs_idx, :, tvs_dim]), np.min(tvs_list[tvs_idx, :, tvs_dim])])

            for tvs_dim in range(cfg.AUDIO.TVS_DIM):
                plt.subplot(cfg.AUDIO.TVS_DIM, sub_num, sub_num * tvs_dim + sub_num)
                plt.plot(range(len(tvs_list[0])), tvs_list[0][:, tvs_dim], color=color_list[0])
                plt.plot(range(len(tvs_list[1])), tvs_list[1][:, tvs_dim], color=color_list[1])
                # plt.yticks([-1, 1])
                if tvs_dim + 1 == cfg.AUDIO.TVS_DIM:
                    plt.xlabel('Frame index')
                if tvs_dim + 1 != cfg.AUDIO.TVS_DIM:
                    plt.xticks([])
            plt.show()
            # show the overlap tvs in big figure
            fig = plt.figure(figsize=(15, 10))
            sub_num = 1
            for tvs_dim in range(cfg.AUDIO.TVS_DIM):
                plt.subplot(cfg.AUDIO.TVS_DIM, sub_num, sub_num * tvs_dim + sub_num)
                plt.plot(range(len(tvs_list[0])), tvs_list[0][:, tvs_dim], color=color_list[0])
                plt.plot(range(len(tvs_list[1])), tvs_list[1][:, tvs_dim], color=color_list[1])
                # plt.yticks([-1, 1])
                if tvs_dim + 1 == cfg.AUDIO.TVS_DIM:
                    plt.xlabel('Frame index')
                if tvs_dim + 1 != cfg.AUDIO.TVS_DIM:
                    plt.xticks([])
            plt.show()
    
    def plot_and_compare_tvs_with_tag_local(self, tvs_list, tag_show=None):
        '''
        tag_show: [x_idx_list, x_tag_list]
        '''
        assert(len(tvs_list) == 2)
        sub_num = 1
        color_list  = ['green', 'red']
        fig = plt.figure(figsize=(15, 10))
        for tvs_dim in range(cfg.AUDIO.TVS_DIM):
            plt.subplot(cfg.AUDIO.TVS_DIM, sub_num, sub_num * tvs_dim + sub_num)
            plt.plot(range(0, len(tvs_list[0]) , 1), tvs_list[0][:, tvs_dim], color=color_list[0])
            plt.plot(range(0, len(tvs_list[1]) , 1), tvs_list[1][:, tvs_dim], color=color_list[1])
            plt.yticks([-1, 1])
            plt.ylabel(self.tvs_label[tvs_dim], rotation=0, verticalalignment='center', size=10)
            if tvs_dim + 1 == cfg.AUDIO.TVS_DIM:
                plt.xlabel('frame index', size=12)
            if tvs_dim + 1 != cfg.AUDIO.TVS_DIM:
                plt.xticks([])
            if tag_show != None and tvs_dim == 0:
                ax = plt.gca()
                ax.set_xticks(tag_show[0])
                ax.xaxis.tick_top()
                ax.set_xticklabels(tag_show[1])
                
        plt.show()

    def plot_and_compare_tvs_with_tag_local_2(self, tvs_list, tag_show=None):
        '''
        tag_show: [x_idx_list, x_tag_list]
        '''
        assert(len(tvs_list) == 3)
        sub_num = 1
        color_list  = ['green', 'red', 'blue']
        fig = plt.figure(figsize=(15, 10))
        for tvs_dim in range(cfg.AUDIO.TVS_DIM):
            plt.subplot(cfg.AUDIO.TVS_DIM, sub_num, sub_num * tvs_dim + sub_num)
            plt.plot(range(0, len(tvs_list[0]) , 1), tvs_list[0][:, tvs_dim], color=color_list[0])
            plt.plot(range(0, len(tvs_list[1]) , 1), tvs_list[1][:, tvs_dim], color=color_list[1])
            plt.plot(range(0, len(tvs_list[2]) , 1), tvs_list[2][:, tvs_dim], color=color_list[2])
            plt.yticks([-1, 1])
            plt.ylabel(self.tvs_label[tvs_dim], rotation=0, verticalalignment='center', size=10)
            if tvs_dim + 1 == cfg.AUDIO.TVS_DIM:
                plt.xlabel('frame index', size=12)
            if tvs_dim + 1 != cfg.AUDIO.TVS_DIM:
                plt.xticks([])
            if tag_show != None and tvs_dim == 0:
                ax = plt.gca()
                ax.set_xticks(tag_show[0])
                ax.xaxis.tick_top()
                ax.set_xticklabels(tag_show[1])
                
        plt.show()

    # tvs_list: [tvs_num, seqlength, tvs_dim]
    def plot_tvs(self, tvs_list):
        color_list = ['green', 'red']
        tvs_num = len(tvs_list) if isinstance(tvs_list, list) else tvs_list.shape[0]
        assert(tvs_num == 1 or tvs_num == 2)
        if tvs_num == 1: sub_num = 1
        if tvs_num == 2: sub_num = 3
        fig = matplotlib.figure.Figure(figsize=(5 * sub_num, 8))

        for tvs_idx in range(tvs_num):
            for tvs_dim in range(cfg.AUDIO.TVS_DIM):
                ax = fig.add_subplot(cfg.AUDIO.TVS_DIM, sub_num, sub_num * tvs_dim + tvs_idx + 1)
                ax.plot(range(len(tvs_list[tvs_idx])), tvs_list[tvs_idx][:, tvs_dim], color=color_list[tvs_idx])
                # ax.set_yticks([])
                # ax.set_yticks([np.max(tvs_list[tvs_idx, :, tvs_dim]), np.min(tvs_list[tvs_idx, :, tvs_dim])])
        if sub_num == 3:
            for tvs_dim in range(cfg.AUDIO.TVS_DIM):
                ax = fig.add_subplot(cfg.AUDIO.TVS_DIM, sub_num, sub_num * tvs_dim + sub_num)
                ax.plot(range(len(tvs_list[0])), tvs_list[0][:, tvs_dim], color=color_list[0])
                ax.plot(range(len(tvs_list[1])), tvs_list[1][:, tvs_dim], color=color_list[1])
                # ax.set_yticks([])
                # ax.set_yticks([np.max(tvs_list[tvs_idx, :, tvs_dim]), np.min(tvs_list[tvs_idx, :, tvs_dim])])

        return fig

    def plot_and_save_tvs(self, tvs_ori, tvs_pred, sequence_lengths, save_num, save_dir):
        tvs_num = 2
        sub_num = 3
        color_list  = ['green', 'red']
        for i in range(save_num):
            _tvs_ori = tvs_ori[i, :sequence_lengths[i], :]
            _tvs_pred = tvs_pred[i, :sequence_lengths[i], :]
            tvs_list = [_tvs_ori, _tvs_pred]
            fig = plt.figure(figsize=(15, 8))
            for tvs_idx in range(tvs_num):
                for tvs_dim in range(cfg.AUDIO.TVS_DIM):
                    plt.subplot(cfg.AUDIO.TVS_DIM, sub_num, sub_num * tvs_dim + tvs_idx + 1)
                    plt.plot(range(len(tvs_list[tvs_idx])), tvs_list[tvs_idx][:, tvs_dim], color_list[tvs_idx])
                    tick_params(direction='in')
                    # plt.ylim(-1, 1)
                    plt.yticks(np.arange(-1, 1, 1))
                    if tvs_idx == 0:
                        plt.ylabel(self.tvs_label[tvs_dim], rotation=0, verticalalignment='center')
                    if tvs_dim + 1 == cfg.AUDIO.TVS_DIM:
                        plt.xlabel('Frame index')
                    if tvs_dim + 1 != cfg.AUDIO.TVS_DIM:
                        plt.xticks([])
                    # ax.set_yticks([])
                    # ax.set_yticks([np.max(tvs_list[tvs_idx, :, tvs_dim]), np.min(tvs_list[tvs_idx, :, tvs_dim])])
            for tvs_dim in range(cfg.AUDIO.TVS_DIM):
                plt.subplot(cfg.AUDIO.TVS_DIM, sub_num, sub_num * tvs_dim + sub_num)
                plt.plot(range(len(tvs_list[0])), tvs_list[0][:, tvs_dim], color=color_list[0])
                plt.plot(range(len(tvs_list[1])), tvs_list[1][:, tvs_dim], color=color_list[1])
                plt.yticks(np.arange(-1, 1, 1))
                if tvs_dim + 1 == cfg.AUDIO.TVS_DIM:
                    plt.xlabel('Frame index')
                if tvs_dim + 1 != cfg.AUDIO.TVS_DIM:
                    plt.xticks([])
            figname = os.path.join(save_dir, 'a2t_tvs_test_{}.png'.format(i))
            print('Save to {}...'.format(figname))
            plt.savefig(figname)
            plt.close()

    def plot_track_local(self, track_list, label_list):
        '''
        plot and show track
        input can be either a track or two tracks, but must be list
        '''
        assert(len(track_list) == len(label_list))
        track_label = cfg.AUDIO.TRACK_DIM_NAME
        color_list = ['green', 'red']
        fig = plt.figure(figsize=(10, 6))
        for i in range(len(track_list)):
            _track = track_list[i]
            plt.plot(range(len(_track)), _track, color=color_list[i], label=label_list[i])
        tick_params(direction='in')
        plt.yticks(np.arange(-1, 1, 1))
        plt.ylabel('Relative magnitude', verticalalignment='center')
        plt.xticks(np.arange(len(track_list[0])), track_label, rotation=30)
        plt.legend(loc='upper right')
        plt.show()

    def plot_and_compare_track(self, track_list):
        '''
        track_list: [ori_track, pred_track], in which both tracks are 1-D vectors
        return a fig
        '''
        track_num = len(track_list) if isinstance(track_list, list) else track_list.shape[0]
        assert(track_num == 2)
        fig = matplotlib.figure.Figure(figsize=(5, 8))
        ax = fig.add_subplot(111)
        ax.plot(range(len(track_list[0])), track_list[0][:], color = 'green')
        ax.plot(range(len(track_list[1])), track_list[1][:], color = 'red')
        return fig

    def plot_and_save_track(self, track_ori, track_pred, save_num, save_dir):
        track_label = cfg.AUDIO.TRACK_DIM_NAME
        for i in range(save_num):
            _track_ori = track_ori[i, :]
            _track_pred = track_pred[i, :]
            fig = plt.figure(figsize=(10, 6))
            plt.plot(range(len(_track_ori)), _track_ori, color='green')
            plt.plot(range(len(_track_pred)), _track_pred, color='red')
            tick_params(direction='in')
            plt.yticks(np.arange(-1, 1, 1))
            plt.ylabel('Relative magnitude', verticalalignment='center')
            plt.xticks(np.arange(len(_track_ori)), track_label, rotation=30)

            figname = os.path.join(save_dir, 'a2t_track_test_{}.png'.format(i))
            print('Save to {}...'.format(figname))
            plt.savefig(figname)
            plt.close()