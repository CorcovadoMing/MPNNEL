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
        input_channels, mpnn_tokens, hidden_channels, tb_size,
        bias = True, stride = 1):

        super().__init__()

        ## Input/output interfaces
        self.input_channels  = input_channels
        self.mpnn_tokens = mpnn_tokens
        self.hidden_channels = hidden_channels
        self.stride = stride
        self.tb_size = tb_size
        
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
        
        self.evw = nn.Parameter(torch.randn(tb_size, self.mpnn_tokens, self.mpnn_tokens))
        self.evw_norm = nn.InstanceNorm2d(tb_size)
        
        self.select_p = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels//8),
            nn.ReLU(),
            nn.Linear(hidden_channels//8, tb_size),
            nn.Sigmoid()
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

        imp = F.softmax(self.select_p(inputs.mean(0)), -1)[..., None, None] # b, tb_size, 1, 1  
        evw = self.evw_norm(self.evw[:, None, ...])[:, 0, ...]
        evw = evw.repeat(imp.size(0), 1, 1, 1) # b, tb_size, self.mpnn_tokens, self.mpnn_tokens
        evw = (evw * imp.expand_as(evw)).sum(1) # b, self.mpnn_tokens, self.mpnn_tokens
        evw = evw.repeat(1, 8, 1, 1)
        
        inputs_norm = self.m_norm1(inputs)
        mv = self.m_f(inputs_norm, inputs_norm, inputs_norm, attn_mask=evw.reshape(-1, self.mpnn_tokens, self.mpnn_tokens))[0] + inputs
        mv = self.m_ffn(mv) + mv
        
        mv_in = torch.cat([mv, inputs], dim=-1)
        mv_in_norm = self.u_norm(mv_in)
        hv = self.u_f(mv_in_norm, mv_in_norm, mv_in_norm)[0] + mv_in
        hv = self.u_ffn(hv) + hv
        
        self.hidden_state = torch.tanh(self.agg(hv))
        
        out_hv = self.scale(self.hidden_state)
        out_hv_norm = self.r_norm(out_hv)
        out = self.r_f(out_hv_norm, out_hv_norm, out_hv_norm)[0] + out_hv
        out = self.r_ffn(out) + out
        
        return out.mean(0), out.mean(0), out.mean(0)


# =============================================================================================

class MPNNELTB(nn.Module):
    def __init__(self,
        input_channels,
        mpnn_tokens,
        hidden_channels,
        tb_size,
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
                                                                   tb_size = tb_size,
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
            queue = [] # previous outputs for skip connection
            output_ = [] # collect the outputs from different layers
            for b in range(self.num_blocks):
                for l in range(self.layers_per_block[b]):
                    lid = "b{}l{}".format(b, l) # layer ID
                    input_, ht, st = self.layers[lid](input_, first_step = (t == 0), mask = length_mask)
                    
                output_.append(input_)
                queue.append(input_)
                
            outputs[t] = output_[-1]
                               
            if length is not None:
                out = torch.zeros(original_batch_size, *outputs[t].shape[1:], device=outputs[t].device, dtype=outputs[t].dtype)
                out[length_mask] = outputs[t]
                outputs[t] = out

        # return the last output_frames of the outputs
        outputs = outputs[-output_frames:]
        
        # 5-th order tensor of size [batch_size, output_frames, channels, height, width]
        outputs = torch.stack(outputs, dim = 1)

        return outputs
