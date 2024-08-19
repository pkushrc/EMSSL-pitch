# coding=utf-8
import sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

sys.path.append('..')
sys.path.append('../misc/')
from misc.config import cfg
from torchsummary import summary

_TINY = 1e-8

def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def transpose_conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1):
        super(BasicResBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class Audio2Tvs(nn.Module):
    def __init__(self):
        super(Audio2Tvs, self).__init__()
        self.spec_dim = cfg.AUDIO.FBANK_NUM
        self.tvs_dim = cfg.AUDIO.TVS_DIM - cfg.AUDIO.TVS_DROP_R0 # 16 - 1, if drop_r0  
        self.track_dim = cfg.AUDIO.TRACK_DIM

        self.lstm_hidden_size = 128
        self.lstm_layers = 1
        self.drop_out = 0

        self.tvs_mseloss = TvsMseLoss()
        self.track_loss = TrackLoss()

        # Operate on Spectrogram
        def _make_conv_layers(module: nn.Sequential, name, in_channels, out_channels, stride):
            downsample = None
            norm_layer = nn.BatchNorm2d
            if stride != 1 or in_channels != out_channels:
                downsample = nn.Sequential(conv1x1(in_channels, out_channels, stride),
                                           norm_layer(out_channels),)
            module.add_module(name, BasicResBlock(in_channels, out_channels, stride=stride, downsample=downsample))

        def _make_deconv_layers(module: nn.Sequential, name, in_channels, out_channels, stride):
            module.add_module(name, transpose_conv3x3(in_channels, out_channels, stride=stride))
            module.add_module(name + '_bn', nn.BatchNorm2d(out_channels))
            module.add_module(name + '_relu', nn.ReLU())

        def _make_dense_layers(module: nn.Module, name, input_dim, output_dim):
            module.add_module(name, nn.Linear(input_dim, output_dim))
            module.add_module(name+'_bn', nn.BatchNorm1d(output_dim))

        self.conv1 = nn.Sequential()
        _make_conv_layers(self.conv1, 'spec_conv_1_1', 1, 64, stride=(2, 2)) 
        _make_conv_layers(self.conv1, 'spec_conv_1_2', 64, 32, stride=(1, 2)) 

        self.conv2 = nn.Sequential()
        _make_conv_layers(self.conv2, 'spec_conv_2', 32, 64, stride=(2, 2)) 
        
        self.conv3 = nn.Sequential()
        _make_conv_layers(self.conv3, 'spec_conv_3', 64, 128, stride=(2, 2)) 

        self.deconv1 = nn.Sequential()
        _make_deconv_layers(self.deconv1, 'spec_deconv_1', 128, 64, stride=(2, 2))

        self.deconv2 = nn.Sequential()
        _make_deconv_layers(self.deconv2, 'spec_deconv_2', 64 + 64, 32, stride=(2, 2))

        self.deconv3 = nn.Sequential()
        _make_deconv_layers(self.deconv3, 'spec_conv_4', 32 + 32, 64, stride=(2, 1))
        
        # To directly flatten parameters, not add to sequence, should unify later
        # def _make_rnn_layers(module: nn.Module, name, input_dim, hidden_dim):
        #     module.add_module(name, nn.LSTM(input_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True))
        # self.blstm = nn.Sequential()
        # _make_rnn_layers(self.blstm, 'spec_lstm_1', self.tvs_dim * 64, self.lstm_hidden_size)
        self.blstm = nn.LSTM(int(self.spec_dim / 4) * 64, self.lstm_hidden_size, num_layers=self.lstm_layers, bidirectional=True, batch_first=True, dropout=self.drop_out)
        # self.blstm_track = nn.LSTM(int(self.spec_dim / 8) * 64, self.lstm_hidden_size, num_layers=self.lstm_layers, bidirectional=True, batch_first=True, dropout=self.drop_out)

        self.linear_tvs = nn.Sequential()
        _make_dense_layers(self.linear_tvs, 'tvs_dense', 2 * self.lstm_hidden_size, self.tvs_dim)
        self.linear_track = nn.Sequential()
        _make_dense_layers(self.linear_track, 'track_dense', 2 * self.lstm_layers * self.lstm_hidden_size, self.track_dim)

    def forward(self, batch_spec: torch.Tensor, seq_length: torch.Tensor):
        """
        Forward Propagation
        :param batch_spec: shape = [batch_size, max_frame_num, mel_bins, 1], i.e. [B, H, W, C], so we need to permute it
        :param batch_tvs: shape = [batch_size, max_frame_num, tvs_dim]
        :param batch_track: shape = [batch_size, track_dim]
        :param sequence_lengths:
        :param training:
        :return:
        TODO: move some of operation to model definition
        """
        # convs
        batch_spec = batch_spec.permute(0, 3, 1, 2)  # to (B, C, H, W) # h, 80
        ntime_steps = batch_spec.shape[2]
        conv1 = self.conv1(batch_spec) # h/2, 20
        conv2 = self.conv2(conv1) # h/4, 10
        conv3 = self.conv3(conv2) # h/8, 5

        # deconvs
        deconv1 = self.deconv1(conv3) # h/4, 10
        deconv1 = F.interpolate(deconv1, size=(conv2.shape[2], conv2.shape[3]), mode='bilinear', align_corners=False)
        deconv2 = self.deconv2(torch.cat((conv2, deconv1), dim=1)) # h/2 20
        deconv2 = F.interpolate(deconv2, size=(conv1.shape[2], conv1.shape[3]), mode='bilinear', align_corners=False)
        deconv3 = self.deconv3(torch.cat((conv1, deconv2), dim=1)) # h, 20
        deconv3 = F.interpolate(deconv3, size=(ntime_steps, int(self.spec_dim / 4)), mode='bilinear', align_corners=False)

        # blstm
        feature_dim = deconv3.shape[1] * deconv3.shape[3]
        deconv3 = deconv3.permute(0, 2, 1, 3).reshape(-1, ntime_steps, feature_dim)  # (B, max_frame_num, 64 * 20)
        self.blstm.flatten_parameters()

        packed_deconv3 = pack_padded_sequence(deconv3, seq_length, batch_first=True, enforce_sorted=False)
        blstm_out_packed, (h_t, c_t) = self.blstm(packed_deconv3) #h_0, c_0 default to be 0
        blstm_out, output_seq_lengths = pad_packed_sequence(blstm_out_packed, batch_first=True, total_length=ntime_steps) #data parallel need to set total_length to make shape of unpacked lstm_out compatiable

        # get prediction of tvs
        blstm_out_flat = blstm_out.reshape(-1, 2 * self.lstm_hidden_size)  # (B x max_frame_num, 2 * lstm_hidden_dim)
        tvs_pred = self.linear_tvs(blstm_out_flat)  # (B x max_frame_num, self.tvs_dim)
        tvs_pred = tvs_pred.reshape(-1, ntime_steps, self.tvs_dim)  # (B, max_frame_num, self.tvs_dim)
        tvs_pred = torch.tanh(tvs_pred)  # (B, max_frame_num, self.tvs_dim)

        # get prediction of track by c_t
        c_t = c_t.permute(1, 0, 2).reshape(-1, 2 * self.lstm_layers * self.lstm_hidden_size)  # (B, 2 * lstm_hidden_dim)
        track_pred = self.linear_track(c_t)  # (B, self.track_dim)
        track_pred = torch.tanh(track_pred)  # (B, self.track_dim)

        return tvs_pred, track_pred


def get_sequence_mask(sequence_lengths, max_len=None):
    '''
    convert [2, 1, 3] into [[1, 1, 0],
                            [1, 0, 0],
                            [1, 1, 1]]   
    max_len: The len to specifiy, >= max(sequence_lengths)
    '''
    if max_len == None:
        max_len = sequence_lengths.max()
    mask_ = torch.arange(max_len).unsqueeze(0).cpu() < sequence_lengths.unsqueeze(1).cpu() # can't run '<' on cuda
    if torch.cuda.is_available():
        mask_ = mask_.cuda()
    return mask_


class TvsMseLoss(torch.nn.Module):
    def __init__(self):
        super(TvsMseLoss, self).__init__()

    def forward(self, tvs_target, tvs_pred, seq_length):
        """
        Masked Mean Square Error
        :param tvs_pred: (B, max_frame_num, self.tvs_dim)
        :param tvs_target: (B, max_frame_num, self.tvs_dim)
        :param seq_length: (B) tensor contains length of each sample
        :return:
        """
        seq_mask = get_sequence_mask(seq_length, max_len=tvs_pred.shape[1])
        # # Mask * (1 / N) * sum(y_hat - y)^2
        # loss = torch.sum(seq_mask * torch.mean((tvs_target - tvs_pred) ** 2, dim=-1)) / torch.sum(seq_mask)
        
        loss = torch.sum(seq_mask * torch.mean((tvs_target - tvs_pred) ** 2, dim=-1)) / seq_mask.numel() # original choice, before 200427
        
        # loss = torch.sum(seq_mask * torch.mean((tvs_target - tvs_pred) ** 2, dim=-1), dim=-1) / seq_length
        # loss = torch.mean(loss) # mean on the batch dim
        return loss


class TrackLoss(torch.nn.Module):
    def __init__(self):
        super(TrackLoss, self).__init__()
        self.loss_func = nn.MSELoss(reduction='mean')
        # self.loss_func = nn.L1Loss(reduction='mean')
    
    def forward(self, track, track_pred):
        return self.loss_func(track, track_pred)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test = Audio2Tvs().to(device)
    summary(test, (120, 80, 1)) # 120 is just for test,
    # x = torch.randn((10, 120, 64, 1)).to(device)
    # tvs_pred, track_pred = test(x)
    # print(tvs_pred.shape, track_pred.shape)
