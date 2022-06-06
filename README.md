# Unified Recurrence Modeling for Video Action Anticipation (MPNNEL)

This is the official PyTorch implementation of [Unified Recurrence Modeling for Video Action Anticipation](https://arxiv.org/abs/2206.01009).


---

## Usage

### MPNNEL (Implicit)
```python
import torch
from mpnnel import MPNNEL

model = MPNNEL(
    input_channels=256,
    mpnn_tokens=8*8,
    hidden_channels=512,
)

input = torch.randn(1, 8, 256, 8, 8) # (Batch, Timesteps, Channels, tokens)
out = model(input)  # (1, 8, 512) -- (Batch, Timesteps, hiden_channels)
```

### MPNNEL-CTP
```python
import torch
from mpnnel_ctp import MPNNELCTP

model = MPNNELCTP(
    input_channels=256,
    mpnn_tokens=8*8,
    hidden_channels=512,
)

input = torch.randn(1, 8, 256, 8, 8) # (Batch, Timesteps, Channels, tokens)
out, noun, verb = model(input)  # (1, 8, 512), (1, 8, 512), (1, 8, 512) -- (Batch, Timesteps, hiden_channels) for out, noun, verb
```

### MPNNEL-TB
```python
import torch
from mpnnel_tb import MPNNELTB

model = MPNNELTB(
    input_channels=256,
    mpnn_tokens=8*8,
    hidden_channels=512,
    tb_size=512
)

input = torch.randn(1, 8, 256, 8, 8) # (Batch, Timesteps, Channels, tokens)
out = model(input)  # (1, 8, 512) -- (Batch, Timesteps, hiden_channels)
```


## Citations

```bibtex
@article{tai2022unified,
  title={Unified Recurrence Modeling for Video Action Anticipation},
  author={Tai, Tsung-Ming and Fiameni, Giuseppe and Lee, Cheng-Kuang and See, Simon and Lanz, Oswald},
  journal={arXiv:2206.01009},
  year={2022}
}
```
