import torch
import torch.nn as nn
import torch.nn.functional as F
import opt_einsum as oe

from torch import nn
import torch
from einops import rearrange, repeat



# ============================================================================================

class MPNNELCell(nn.Module):
    def __init__(self,
        input_channels, mpnn_tokens, hidden_channels,
        bias = True, stride = 1):

        super().__init__()

        ## Input/output interfaces
        self.input_channels  = input_channels
        self.mpnn_tokens = mpnn_tokens
        self.hidden_channels = hidden_channels
        self.stride = stride
        
        self.encoder = nn.Sequential(
            nn.Linear(input_channels, hidden_channels, bias=False),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(inplace=True),
        )
        
        self.add_h = nn.Sequential(
            nn.Linear(2*hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
        )
        
        self.m_norm1 = nn.LayerNorm(hidden_channels)
        self.m_norm2 = nn.LayerNorm(hidden_channels)
        self.m_f = nn.MultiheadAttention(hidden_channels, 8)
        self.m_ffn = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        
        self.u_norm = nn.LayerNorm(2*hidden_channels)
        self.u_f = nn.MultiheadAttention(2*hidden_channels, 8)
        self.u_ffn = nn.Sequential(
            nn.LayerNorm(2*hidden_channels),
            nn.Linear(2*hidden_channels, 2*hidden_channels),
            nn.GELU(),
            nn.Linear(2*hidden_channels, 2*hidden_channels),
        )
        
        self.r_norm = nn.LayerNorm(hidden_channels)
        self.r_f = nn.MultiheadAttention(hidden_channels, 8)
        self.r_ffn = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        
        self.agg = nn.Sequential(
            nn.Linear(2*hidden_channels, hidden_channels),
        )
        self.scale = nn.Linear(hidden_channels, hidden_channels)
        
        self.spatial_filter = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels//8),
            nn.ReLU(),
            nn.Linear(hidden_channels//8, 1),
            nn.Sigmoid()
        )
        
        self.v_proposal = nn.Parameter(torch.randn(1, self.hidden_channels))
        self.n_proposal = nn.Parameter(torch.randn(1, self.hidden_channels))
        
        self.v_t = nn.Sequential(
            nn.Linear(hidden_channels, self.mpnn_tokens),
            nn.LayerNorm(self.mpnn_tokens),
        )
        self.n_t = nn.Sequential(
            nn.Linear(hidden_channels, self.mpnn_tokens),
            nn.LayerNorm(self.mpnn_tokens),
        )
        
        
    def initialize(self, inputs):
        device = inputs.device # "cpu" or "cuda"
        dtype = inputs.dtype
        batch_size, _, height, width = inputs.size()
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.hidden_state = torch.zeros(self.mpnn_tokens, batch_size, self.hidden_channels, device=inputs.device, dtype=inputs.dtype)
        
    
    def _new_state(self, x, mask):
        new_state = torch.zeros(self.batch_size, *x.shape[1:], device=x.device, dtype=x.dtype)
        new_state[mask] = x
        return new_state

        
    def forward(self, inputs, first_step = False, checkpointing = False, mask = None):
        if first_step: self.initialize(inputs) # intialize the states at the first step
            
        org_shape = inputs.shape
            
        inputs = inputs.reshape(*inputs.shape[:2], -1).transpose(-1, -2)
        inputs = self.encoder(inputs)
        inputs = self.add_h(torch.cat([inputs, self.hidden_state.transpose(0, 1)], dim=-1))
        
        # spatial filter
        sf = self.spatial_filter(inputs)
        sf = sf.expand_as(inputs)
        inputs = inputs * sf
        
        inputs = inputs.transpose(0, 1) # self.mpnn_tokens, b, 1024

        vt = self.v_t(self.v_proposal)
        nt = self.n_t(self.n_proposal)
        evw = torch.einsum('bi,bj->ij', vt, nt)
        hevw = torch.einsum('vw,wbc->vbc', torch.softmax(evw, dim=-1), inputs)
        
        inputs_norm = self.m_norm1(inputs)
        hevw_norm = self.m_norm2(hevw)
        mv = self.m_f(inputs_norm, hevw_norm, inputs_norm)[0] + inputs
        mv = self.m_ffn(mv) + mv
        
        mv_in = torch.cat([mv, inputs], dim=-1)
        mv_in_norm = self.u_norm(mv_in)
        hv = self.u_f(mv_in_norm, mv_in_norm, mv_in_norm)[0] + mv_in
        hv = self.u_ffn(hv) + hv
        
        self.hidden_state = torch.tanh(self.agg(hv))
        
        vt = self.v_proposal[:, None, :].repeat(1, mv.size(1), 1)
        nt = self.n_proposal[:, None, :].repeat(1, mv.size(1), 1)
        out_hv = self.scale(self.hidden_state)
        out_hv = torch.cat([vt, nt, out_hv], dim=0)
        out_hv_norm = self.r_norm(out_hv)
        out = self.r_f(out_hv_norm, out_hv_norm, out_hv_norm)[0] + out_hv
        out = self.r_ffn(out) + out
        
        vt = out[0]
        nt = out[1]
        out = out[2:]
        
        return out.mean(0), nt, vt

# =============================================================================================

class MPNNELCTP(nn.Module):
    def __init__(self,
        input_channels,
        mpnn_tokens,
        hidden_channels,
        bias = True):
        
        super().__init__()

        ## Hyperparameters
        self.layers_per_block = [1]
        self.hidden_channels  = hidden_channels
        
        self.num_blocks = len(self.layers_per_block)


        Cell = lambda in_channels, out_channels, stride: MPNNELCell(
                                                                   input_channels = in_channels, 
                                                                   mpnn_tokens = mpnn_tokens,
                                                                   hidden_channels = out_channels,
                                                                   bias = bias, 
                                                                   stride = stride
                                                                  )

        ## Construction of convolutional tensor-train LSTM network

        # stack the convolutional-LSTM layers with skip connections 
        self.layers = nn.ModuleDict()
        for b in range(self.num_blocks):
            for l in range(self.layers_per_block[b]):
                # number of input channels to the current layer
                if l > 0: 
                    channels = hidden_channels
                elif l == 0 and b == 0:
                    channels = input_channels
                elif l == 0 and b > 0:
                    channels = hidden_channels[b-1]
                    if self.input_shortcut:
                        channels += input_channels
                        
                lid = "b{}l{}".format(b, l) # layer ID
                self.layers[lid] = Cell(channels, hidden_channels, 1)
            

    def forward(self, inputs, length = None, input_frames = None, future_frames=0, output_frames = None):
        if input_frames is None:
            input_frames = inputs.size(1)
        
        if output_frames is None:
            output_frames = input_frames

        total_steps = input_frames + future_frames
        outputs = [None] * total_steps
        outputs_ht = [None] * total_steps
        outputs_st = [None] * total_steps

        for t in range(total_steps):
            # input_: 4-th order tensor of size [batch_size, input_channels, height, width]
            if t < input_frames: 
                input_ = inputs[:, t]
            
            # length-aware
            if length is None:
                length = torch.stack([torch.LongTensor([total_steps]) for _ in range(input_.size(0))]).to(input_.device).squeeze(-1)
                
            original_batch_size = input_.size(0)
            length_mask = length>t
            input_ = input_[length_mask]                

            backup_input = input_
            output_ = [] # collect the outputs from different layers
            output_ht_ = []
            output_st_ = []
            for b in range(self.num_blocks):
                for l in range(self.layers_per_block[b]):
                    lid = "b{}l{}".format(b, l) # layer ID
                    input_, ht, st = self.layers[lid](input_, first_step = (t == 0), mask = length_mask)
                    
                output_.append(input_)
                output_ht_.append(ht)
                output_st_.append(st)
                
            outputs[t] = output_[-1]
            outputs_ht[t] = output_ht_[-1]
            outputs_st[t] = output_st_[-1]
                               
            if length is not None:
                out = torch.zeros(original_batch_size, *outputs[t].shape[1:], device=outputs[t].device, dtype=outputs[t].dtype)
                out[length_mask] = outputs[t]
                outputs[t] = out

        # return the last output_frames of the outputs
        outputs = outputs[-output_frames:]
        outputs_ht = outputs_ht[-output_frames:]
        outputs_st = outputs_st[-output_frames:]
        
        # 5-th order tensor of size [batch_size, output_frames, channels, height, width]
        outputs = torch.stack(outputs, dim = 1)
        outputs_ht = torch.stack(outputs_ht, dim = 1)
        outputs_st = torch.stack(outputs_st, dim = 1)

        return outputs, outputs_ht, outputs_st
