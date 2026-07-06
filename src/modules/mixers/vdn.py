"""
VDN (Value Decomposition Network) Mixer
========================================
Simplest value decomposition: Q_tot = sum(Q_i)
No parameters, no state conditioning.

Reference: Sunehag et al., "Value-Decomposition Networks For Cooperative
Multi-Agent Learning", 2017.
"""
import torch as th
import torch.nn as nn


class VDNMixer(nn.Module):
    def forward(self, agent_qs, batch):
        return th.sum(agent_qs, dim=2, keepdim=True)
