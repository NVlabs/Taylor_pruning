"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from torch.optim.optimizer import Optimizer

PRINT_ALL = False
USE_FULL = False

class group_lasso_decay(Optimizer):
    r"""Implements group lasso weight decay (GLWD) that pushed entire group to 0.
    Normal weight decay makes weight sparse, GLWD will make sparse channel.
    Assumes we want to decay the group related to feature maps (channels), other groups are possible but not implemented
    """

    def __init__(self, params, group_lasso_weight=0, named_parameters=None, output_sizes=None):
        defaults = dict(group_lasso_weight = group_lasso_weight, total_neurons = 0)

        super(group_lasso_decay, self).__init__(params, defaults)

        self.per_layer_per_neuron_stats = {'flops': list(), 'params': list(), 'latency': list()}

        self.named_parameters = named_parameters

        self.output_sizes = None
        if output_sizes is not None:
            self.output_sizes = output_sizes

    def __setstate__(self, state):
        super(group_lasso_decay, self).__setstate__(state)

    def get_number_neurons(self, print_output = False):
        total_neurons = 0
        for gr_ind, group in enumerate(self.param_groups):
            total_neurons += group['total_neurons'].item()
        # if print_output:
        #     print("Total parameters: ",total_neurons, " or ", total_neurons/1e7, " times 1e7")
        return total_neurons

    def get_number_flops(self, print_output = False):
        total_flops = 0
        for gr_ind, group in enumerate(self.param_groups):
            total_flops += group['total_flops'].item()

        total_neurons = 0
        for gr_ind, group in enumerate(self.param_groups):
            total_neurons += group['total_neurons'].item()

        if print_output:
            # print("Total flops: ",total_flops, " or ", total_flops/1e9, " times 1e9")

            print("Flops 1e 9/params 1e7:  %3.3f &  %3.3f"%(total_flops/1e9, total_neurons/1e7))
        return total_flops

    def step(self, closure=None):
        """Applies GLWD regularization to weights.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            group_lasso_weight = group['group_lasso_weight']
            group['total_neurons'] = 0
            group['total_flops'] = 0

            for p in group['params']:
                param_state = self.state[p]

                if 1:

                    weight_size = p.data.size()

                    if ('group_lasso_coeff' not in param_state) or (param_state['group_lasso_coeff'].shape != p.data.shape):
                        nunits = p.data.size(0)
                        param_state['group_lasso_coeff'] = p.data.clone().view(nunits,-1).sum(dim=1)*0.0 + group_lasso_weight

                    group_lasso_weight_local = param_state['group_lasso_coeff']

                    if len(weight_size) == 4:
                        # defined for conv layers only
                        nunits = p.data.size(0)

                        # let's compute denominator
                        divider = p.data.pow(2).view(nunits,-1).sum(dim=1).pow(0.5)

                        eps = 1e-5
                        eps2 = 1e-13
                        # check if divider is above threshold
                        divider_bool = divider.gt(eps).view(-1).float()

                        group_lasso_gradient = p.data * (group_lasso_weight_local * divider_bool /
                                               (divider + eps2)).view(nunits, 1, 1, 1).repeat(1, weight_size[1],weight_size[2],weight_size[3])

                        # apply weight decay step:
                        p.data.add_(-1.0, group_lasso_gradient)
        return loss

    def step_after(self, closure=None):
        """Computes FLOPS and number of neurons after considering zeroed out input and outputs.
        Channels are assumed to be pruned if their l2 norm is very small or if magnitude of gradient is very small.
        This function does not perform weight pruning, weights are untouched.

        This function also calls push_biases_down which sets corresponding biases to 0.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        param_index = -1
        conv_param_index = -1
        for group in self.param_groups:
            group_lasso_weight = group['group_lasso_weight']
            group['total_neurons'] = 0
            group['total_flops'] = 0

            for p in group['params']:

                if group_lasso_weight != 0 or 1:
                    weight_size = p.data.size()

                    if (len(weight_size) == 4) or (len(weight_size) == 2) or (len(weight_size) == 1):
                        param_index += 1
                        # defined for conv layers only
                        nunits = p.data.size(0)
                        # let's compute denominator
                        divider = p.data.pow(2).view(nunits,-1).sum(dim=1).pow(0.5)

                        eps = 1e-4
                        # check if divider is above threshold
                        divider_bool = divider.gt(eps).view(-1).float()

                        if (len(weight_size) == 4) or (len(weight_size) == 2) or (len(weight_size) == 1):
                            if not (p.grad is None):
                                # consider gradients as well and if gradient is below spesific threshold than we claim parameter to be removed
                                divider_grad = p.grad.data.pow(2).view(nunits, -1).sum(dim=1).pow(0.5)
                                eps = 1e-8
                                divider_bool_grad = divider_grad.gt(eps).view(-1).float()
                                divider_bool = divider_bool_grad * divider_bool

                                if (len(weight_size) == 4) or (len(weight_size) == 2):
                                    # get gradient for input:
                                    divider_grad_input = p.grad.data.pow(2).transpose(0,1).contiguous().view(p.data.size(1),-1).sum(dim=1).pow(0.5)
                                    divider_bool_grad_input = divider_grad_input.gt(eps).view(-1).float()

                                    divider_input = p.data.pow(2).transpose(0,1).contiguous().view(p.data.size(1), -1).sum(dim=1).pow(0.5)
                                    divider_bool_input = divider_input.gt(eps).view(-1).float()
                                    divider_bool_input = divider_bool_input * divider_bool_grad_input
                                    # if gradient is small then remove it out

                        if USE_FULL:
                            # reset to evaluate true number of flops and neurons
                            # useful for full network only
                            divider_bool = 0.0*divider_bool + 1.0
                            divider_bool_input = 0.0*divider_bool_input + 1.0

                        if len(weight_size) == 4:
                            p.data.mul_(divider_bool.view(nunits,1, 1, 1).repeat(1,weight_size[1], weight_size[2], weight_size[3]))
                            current_neurons = divider_bool.sum()*divider_bool_input.sum()*weight_size[2]* weight_size[3]

                        if len(weight_size) == 2:
                            current_neurons = divider_bool.sum()*divider_bool_input.sum()

                        if len(weight_size) == 1:
                            current_neurons = divider_bool.sum()
                            # add mean and var over batches
                            current_neurons = current_neurons + divider_bool.sum()

                        group['total_neurons'] += current_neurons

                        if len(weight_size) == 4:
                            conv_param_index += 1
                            input_channels  = divider_bool_input.sum()
                            output_channels = divider_bool.sum()

                            if self.output_sizes is not None:
                                output_height, output_width = self.output_sizes[conv_param_index][-2:]
                            else:
                                if hasattr(p, 'output_dims'):
                                    output_height, output_width = p.output_dims[-2:]
                                else:
                                    output_height, output_width = 0, 0

                            kernel_ops = weight_size[2] * weight_size[3] * input_channels

                            params = output_channels * kernel_ops
                            flops  = params * output_height * output_height

                            # add flops due to batch normalization
                            flops = flops + output_height * output_width*3

                            group['total_flops'] = group['total_flops'] + flops

                        if len(weight_size) == 1:
                            flops = len(weight_size)
                            group['total_flops'] = group['total_flops'] + flops
                        if len(weight_size) == 2:
                            input_channels  = divider_bool_input.sum()
                            output_channels = divider_bool.sum()
                            flops = input_channels * output_channels
                            group['total_flops'] = group['total_flops'] + flops

                        if len(self.per_layer_per_neuron_stats['flops']) <= param_index:
                            self.per_layer_per_neuron_stats['flops'].append(flops / divider_bool.sum())
                            self.per_layer_per_neuron_stats['params'].append(current_neurons / divider_bool.sum())
                            # self.per_layer_per_neuron_stats['latency'].append(flops / output_channels)
                        else:
                            self.per_layer_per_neuron_stats['flops'][param_index] = flops / divider_bool.sum()
                            self.per_layer_per_neuron_stats['params'][param_index] = current_neurons / divider_bool.sum()
                            # self.per_layer_per_neuron_stats['latency'][param_index] = flops / output_channels

        self.push_biases_down(eps=1e-3)
        return loss

    def push_biases_down(self, eps=1e-3):
        '''
        This function goes over parameters and sets according biases to zero,
        without this function biases will not be zero
        '''
        # first pass
        list_of_names = []
        for name, param in self.named_parameters:
            if "weight" in name:
                weight_size = param.data.shape
                if (len(weight_size) == 4) or (len(weight_size) == 2):
                    # defined for conv layers only
                    nunits = weight_size[0]
                    # let's compute denominator
                    divider = param.data.pow(2).view(nunits, -1).sum(dim=1).pow(0.5)
                    divider_bool = divider.gt(eps).view(-1).float()
                    list_of_names.append((name.replace("weight", "bias"), divider_bool))

        # second pass
        for name, param in self.named_parameters:
            if "bias" in name:
                for ind in range(len(list_of_names)):
                    if list_of_names[ind][0] == name:
                        param.data.mul_(list_of_names[ind][1])





