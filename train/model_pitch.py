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


class Audio2Tvs(nn.Module):
    def __init__(self):
        super(Audio2Tvs, self).__init__()
        self.spec_dim = cfg.AUDIO.FBANK_NUM
        self.tvs_dim = cfg.AUDIO.TVS_DIM - cfg.AUDIO.TVS_DROP_R0 # 16 - 1, if drop_r0  
        self.track_dim = cfg.AUDIO.TRACK_DIM

        self.drop_out = 0
        self.seperator = Separator(self.spec_dim)
        self.linear_tvs = nn.Linear(self.spec_dim,self.tvs_dim)
        self.linear_track = nn.Linear(self.spec_dim,self.track_dim)
        self.loss = MseLoss()
        

       
    def forward(self, batch_spec: torch.Tensor):
        """
        Forward Propagation
        :param batch_spec: shape = [batch_size, max_frame_num, mel_bins, 1], i.e. [B, T, d_model, C], so we need to permute it
        :param batch_tvs: shape = [batch_size, K, tvs_dim]
        :param batch_track: shape = [batch_size,K, track_dim]
        :param sequence_lengths:
        :param training:
        :return:
        TODO: move some of operation to model definition
        """
        # convs
        batch_spec = batch_spec.squeeze(-1) # to (B, T , d_model)
        batch_spec = batch_spec.permute(0, 2, 1)  # to (B, d_mdoel,T) 
        ntime_steps = batch_spec.shape[2]

        sep = self.seperator(batch_spec) #(B, speakers,d_model, T)
        sep = sep.permute(0,1,3,2)#(B, speakers,T,d_model)
        
        tvs_pred = self.linear_tvs(sep)  
        tvs_pred = tvs_pred.reshape(-1,2, ntime_steps, self.tvs_dim)  
        tvs_pred = torch.tanh(tvs_pred) 

        track_pred = self.linear_track(torch.mean(sep,dim = 2))  
        track_pred = track_pred.reshape(-1,2, self.track_dim)
        track_pred = torch.tanh(track_pred)  

        return tvs_pred, track_pred


class Separator(nn.Module):
    def __init__(self, d_model, n_layers = 3, speakers = 2):
        super(Separator, self).__init__();

        self.bottleneck = nn.Sequential(
                                        nn.GroupNorm(1, d_model, eps=1e-16),
                                        nn.Conv1d(d_model, d_model, 1, bias=False),
                                    );
        self.blocks = nn.ModuleList([
            TCNBlock(d_bn = d_model) for _ in range(n_layers)
        ])

        self.output = nn.Sequential(
                                    nn.PReLU(),
                                    nn.Conv1d(d_model, speakers*d_model, 1, bias = False),
                                    nn.ReLU()
        )
        self.d_model = d_model;

    def forward(self, x):
        '''
        TCN network, input encoded feature, output enhanced signals
        Input:
            x: [B, d_model, K];
        Output:
            output: [B, speakers,d_model, K];
        '''
        B, d_model, K = x.size()
        output = self.bottleneck(x);
        skip_connection = 0;
        for block in self.blocks:
            skip, output = block(output);
            skip_connection = skip_connection + skip;
        output = self.output(skip_connection);
        output = output.view(B,2,d_model,K)
        return output;

class TCNBlock(nn.Module):
    def __init__(self,d_bn, n_tcn_layers=8):
        super(TCNBlock, self).__init__()
        self.blocks = nn.ModuleList([
            DepthConv1d(2**x,d_bn) for x in range(n_tcn_layers)
        ])
    
    def forward(self, x):
        '''
        Dialation conv1d block, including X conv1d block with 2^0 to 2^(x-1) dilation
        Input:
            X: [B, d_bn, K];
        Output:
            output: tensor with same shape as input
            skip_connection: [B, d_skip, K]
        '''
        output = x;
        skip_connection = 0;
        for block in self.blocks:
            skip, output = block(output);
            skip_connection = skip + skip_connection;
        return skip_connection, output

class DepthConv1d(nn.Module):
    def __init__(self, dilation, d_bn, d_inside = 128, kernel_size_inside = 3, d_skip = cfg.AUDIO.FBANK_NUM):
        super(DepthConv1d, self).__init__()

        self.net = nn.Sequential(
                                    nn.Conv1d(d_bn, d_inside, 1, bias = False),
                                    nn.PReLU(d_inside), # nn.PReLU(),
                                    # nn.PReLU(), 
                                    nn.GroupNorm(1, d_inside, eps = 1e-16),
                                    nn.Conv1d(d_inside, d_inside, kernel_size_inside, dilation=dilation, groups=d_inside,
                                            padding=(kernel_size_inside-1)*dilation//2, bias=False),
                                    nn.PReLU(d_inside), # nn.PReLU(),
                                    # nn.PReLU(),
                                    nn.GroupNorm(1, d_inside, eps = 1e-16)
                                )

        self.output = nn.Conv1d(d_inside, d_bn,   1, bias = False);
        self.skip   = nn.Conv1d(d_inside, d_skip, 1, bias = False);

    def forward(self, x):
        residual = x;
        x = self.net(x);
        output = self.output(x) + residual;
        skip = self.skip(x);
        return skip, output;



class MseLoss(torch.nn.Module):
    def __init__(self):
        super(MseLoss, self).__init__()
        self.loss_func = nn.MSELoss(reduction='mean')

    def forward(self, tvs_target, tvs_pred, track_target, track_pred):
        
        tvs_loss1 = self.loss_func(tvs_target,tvs_pred) 
        track_loss1 = self.loss_func(track_target,track_pred)

        tvs1 = tvs_pred[:,:1,:,:]
        tvs2 = tvs_pred[:,1:,:,:]
        track1 = track_pred[:,:1,:]
        track2 = track_pred[:,1:,:]
        tvs_pred = torch.cat((tvs2,tvs1),dim=1)
        track_pred = torch.cat((track2,track1),dim=1)

        tvs_loss2 = self.loss_func(tvs_target,tvs_pred) 
        track_loss2 = self.loss_func(track_target,track_pred)

        if tvs_loss1 + track_loss1 < tvs_loss2 + track_loss2:
            return tvs_loss1 , track_loss1
        else:
            return tvs_loss2 , track_loss2
        




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test = Audio2Tvs().to(device)
    x = torch.randn((10, 120, 80, 1)).to(device)
    tvs_pred, track_pred = test(x)
    print(tvs_pred.shape, track_pred.shape)
