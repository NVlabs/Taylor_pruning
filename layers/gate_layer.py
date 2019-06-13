"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""


import torch
import torch.nn as nn

'''
Gating layer for pruning
'''

class GateLayer(nn.Module):
    def __init__(self, input_features, output_features, size_mask):
        super(GateLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.size_mask = size_mask
        self.weight = nn.Parameter(torch.ones(output_features))

        # for simpler way to find these layers
        self.do_not_update = True

    def forward(self, input):
        return input*self.weight.view(*self.size_mask)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.input_features, self.output_features is not None
        )