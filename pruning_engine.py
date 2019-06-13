"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from __future__ import print_function
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

import numpy as np


from copy import deepcopy
import itertools
import pickle
import json

METHOD_ENCODING = {0: "Taylor_weight", 1: "Random", 2: "Weight norm", 3: "Weight_abs",
                   6: "Taylor_output", 10: "OBD", 11: "Taylor_gate_SO",
                   22: "Taylor_gate", 23: "Taylor_gate_FG", 30: "BN_weight", 31: "BN_Taylor"}


# Method is encoded as an integer that mapping is shown above.
# Methods map to the paper as follows:
# 0 - Taylor_weight - Conv weight/conv/linear weight with Taylor FO In Table 2 and Table 1
# 1 - Random        - Random
# 2 - Weight norm   - Weight magnitude/ weight
# 3 - Weight_abs    - Not used
# 6 - Taylor_output - Taylor-output as is [27]
# 10- OBD           - OBD
# 11- Taylor_gate_SO- Taylor SO
# 22- Taylor_gate   - Gate after BN in Table 2, Taylor FO in Table 1
# 23- Taylor_gate_FG- uses gradient per example to compute Taylor FO, Taylor FO- FG in Table 1, Gate after BN - FG in Table 2
# 30- BN_weight     - BN scale in Table 2
# 31- BN_Taylor     - BN scale Taylor FO in Table 2


class PruningConfigReader(object):
    def __init__(self):
        self.pruning_settings = {}
        self.config = None

    def read_config(self, filename):
        # reads .json file and sets values as pruning_settings for pruning

        with open(filename, "r") as f:
            config = json.load(f)

        self.config = config

        self.read_field_value("method", 0)
        self.read_field_value("frequency", 500)
        self.read_field_value("prune_per_iteration", 2)
        self.read_field_value("maximum_pruning_iterations", 10000)
        self.read_field_value("starting_neuron", 0)

        self.read_field_value("fixed_layer", -1)
        # self.read_field_value("use_momentum", False)

        self.read_field_value("pruning_threshold", 100)
        self.read_field_value("start_pruning_after_n_iterations", 0)
        # self.read_field_value("use_momentum", False)
        self.read_field_value("do_iterative_pruning", True)
        self.read_field_value("fixed_criteria", False)
        self.read_field_value("seed", 0)
        self.read_field_value("pruning_momentum", 0.9)
        self.read_field_value("flops_regularization", 0.0)
        self.read_field_value("prune_neurons_max", 1)

        self.read_field_value("group_size", 1)

    def read_field_value(self, key, default):
        param = default
        if key in self.config:
            param = self.config[key]

        self.pruning_settings[key] = param

    def get_parameters(self):
        return self.pruning_settings


class pytorch_pruning(object):
    def __init__(self, parameters, pruning_settings=dict(), log_folder=None):
        def initialize_parameter(object_name, settings, key, def_value):
            '''
            Function check if key is in the settings and sets it, otherwise puts default momentum
            :param object_name: reference to the object instance
            :param settings: dict of settings
            :param def_value: def value for the parameter to be putted into the field if it doesn't work
            :return:
            void
            '''
            value = def_value
            if key in settings.keys():
                value = settings[key]
            setattr(object_name, key, value)

        # store some statistics
        self.min_criteria_value = 1e6
        self.max_criteria_value = 0.0
        self.median_criteria_value = 0.0
        self.neuron_units = 0
        self.all_neuron_units = 0
        self.pruned_neurons = 0
        self.gradient_norm_final = 0.0
        self.flops_regularization = 0.0 #not used in the paper
        self.pruning_iterations_done = 0

        # initialize_parameter(self, pruning_settings, 'use_momentum', False)
        initialize_parameter(self, pruning_settings, 'pruning_momentum', 0.9)
        initialize_parameter(self, pruning_settings, 'flops_regularization', 0.0)
        self.momentum_coeff = self.pruning_momentum
        self.use_momentum = self.pruning_momentum > 0.0

        initialize_parameter(self, pruning_settings, 'prune_per_iteration', 1)
        initialize_parameter(self, pruning_settings, 'start_pruning_after_n_iterations', 0)
        initialize_parameter(self, pruning_settings, 'prune_neurons_max', 0)
        initialize_parameter(self, pruning_settings, 'maximum_pruning_iterations', 0)
        initialize_parameter(self, pruning_settings, 'pruning_silent', False)
        initialize_parameter(self, pruning_settings, 'l2_normalization_per_layer', False)
        initialize_parameter(self, pruning_settings, 'fixed_criteria', False)
        initialize_parameter(self, pruning_settings, 'starting_neuron', 0)
        initialize_parameter(self, pruning_settings, 'frequency', 30)
        initialize_parameter(self, pruning_settings, 'pruning_threshold', 100)
        initialize_parameter(self, pruning_settings, 'fixed_layer', -1)
        initialize_parameter(self, pruning_settings, 'combination_ID', 0)
        initialize_parameter(self, pruning_settings, 'seed', 0)
        initialize_parameter(self, pruning_settings, 'group_size', 1)

        initialize_parameter(self, pruning_settings, 'method', 0)

        # Hessian related parameters
        self.temp_hessian = [] # list to store Hessian
        self.hessian_first_time = True

        self.parameters = list()

        ##get pruning parameters
        for parameter in parameters:
            parameter_value = parameter["parameter"]
            self.parameters.append(parameter_value)

        if self.fixed_layer == -1:
            ##prune all layers
            self.prune_layers = [True for parameter in self.parameters]
        else:
            ##prune only one layer
            self.prune_layers = [False, ]*len(self.parameters)
            self.prune_layers[self.fixed_layer] = True

        self.iterations_done = 0

        self.prune_network_criteria = list()
        self.prune_network_accomulate = {"by_layer": list(), "averaged": list(), "averaged_cpu": list()}

        self.pruning_gates = list()
        for layer in range(len(self.parameters)):
            self.prune_network_criteria.append(list())

            for key in self.prune_network_accomulate.keys():
                self.prune_network_accomulate[key].append(list())

            self.pruning_gates.append(np.ones(len(self.parameters[layer]),))
            layer_now_criteria = self.prune_network_criteria[-1]
            for unit in range(len(self.parameters[layer])):
                layer_now_criteria.append(0.0)

        # logging setup
        self.log_folder = log_folder
        self.folder_to_write_debug = self.log_folder + '/debug/'
        if not os.path.exists(self.folder_to_write_debug):
            os.makedirs(self.folder_to_write_debug)

        self.method_25_first_done = True

        if self.method == 40 or self.method == 50 or self.method == 25:
            self.oracle_dict = {"layer_pruning": -1, "initial_loss": 0.0, "loss_list": list(), "neuron": list(), "iterations": 0}
            self.method_25_first_done = False

        if self.method == 25:
            with open("./utils/study/oracle.pickle","rb") as f:
                oracle_list = pickle.load(f)

            self.oracle_dict["loss_list"] = oracle_list

        self.needs_hessian = False
        if self.method in [10, 11]:
            self.needs_hessian = True

        # useful for storing data of the experiment
        self.data_logger = dict()
        self.data_logger["pruning_neurons"] = list()
        self.data_logger["pruning_accuracy"] = list()
        self.data_logger["pruning_loss"] = list()
        self.data_logger["method"] = self.method
        self.data_logger["prune_per_iteration"] = self.prune_per_iteration
        self.data_logger["combination_ID"] = list()
        self.data_logger["fixed_layer"] = self.fixed_layer
        self.data_logger["frequency"] = self.frequency
        self.data_logger["starting_neuron"] = self.starting_neuron
        self.data_logger["use_momentum"] = self.use_momentum

        self.data_logger["time_stamp"] = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

        if hasattr(self, 'seed'):
            self.data_logger["seed"] = self.seed

        self.data_logger["filename"] = "%s/data_logger_seed_%d_%s.p"%(log_folder, self.data_logger["seed"], self.data_logger["time_stamp"])
        if self.method == 50:
            self.data_logger["filename"] = "%s/data_logger_seed_%d_neuron_%d_%s.p"%(log_folder, self.starting_neuron, self.data_logger["seed"], self.data_logger["time_stamp"])
        self.log_folder = log_folder

        # the rest of initializations
        self.pruned_neurons = self.starting_neuron

        self.util_loss_tracker = 0.0
        self.util_acc_tracker = 0.0
        self.util_loss_tracker_num = 0.0

        self.loss_tracker_exp = ExpMeter()
        # stores results of the pruning, 0 - unsuccessful, 1 - successful
        self.res_pruning = 0

        self.iter_step = -1

        self.train_writer = None

        self.set_moment_zero = True
        self.pruning_mask_from = ""

    def add_criteria(self):
        '''
        This method adds criteria to global list given batch stats.
        '''

        if self.fixed_criteria:
            if self.pruning_iterations_done > self.start_pruning_after_n_iterations :
                return 0

        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            nunits = self.parameters[layer].size(0)
            eps = 1e-8

            if len(self.pruning_mask_from) > 0:
                # preload pruning mask
                self.method = -1
                criteria_for_layer = torch.from_numpy(self.loaded_mask_criteria[layer]).type(torch.FloatTensor).cuda(async=True)

            if self.method == 0:
                # First order Taylor expansion on the weight
                criteria_for_layer = (self.parameters[layer]*self.parameters[layer].grad ).data.pow(2).view(nunits,-1).sum(dim=1)
            elif self.method == 1:
                # random pruning
                criteria_for_layer = np.random.uniform(low=0, high=5, size=(nunits,))
            elif self.method == 2:
                # min weight
                criteria_for_layer = self.parameters[layer].pow(2).view(nunits,-1).sum(dim=1).data
            elif self.method == 3:
                # weight_abs
                criteria_for_layer = self.parameters[layer].abs().view(nunits,-1).sum(dim=1).data
            elif self.method == 6:
                # ICLR2017 Taylor on output of the layer
                if 1:
                    criteria_for_layer = self.parameters[layer].full_grad_iclr2017
                    criteria_for_layer = criteria_for_layer / (np.linalg.norm(criteria_for_layer) + eps)
            elif self.method == 10:
                # diagonal of Hessian
                criteria_for_layer = (self.parameters[layer] * torch.diag(self.temp_hessian[layer])).data.view(nunits,
                                                                                                               -1).sum(
                    dim=1)
            elif self.method == 11:
                #  second order Taylor expansion for loss change in the network
                criteria_for_layer = (-(self.parameters[layer] * self.parameters[layer].grad).data + 0.5 * (
                            self.parameters[layer] * self.parameters[layer] * torch.diag(
                        self.temp_hessian[layer])).data).pow(2)

            elif self.method == 22:
                # Taylor pruning on gate
                criteria_for_layer = (self.parameters[layer]*self.parameters[layer].grad).data.pow(2).view(nunits, -1).sum(dim=1)
                if hasattr(self, "dataset"):
                    # fix for skip connection pruning, gradient will be accumulated instead of being averaged
                    if self.dataset == "Imagenet":
                        if hasattr(self, "model"):
                            if not ("noskip" in self.model):
                                if "resnet" in self.model:
                                    mult = 3.0
                                    if layer == 1:
                                        mult = 4.0
                                    elif layer == 2:
                                        mult = 23.0 if "resnet101" in self.model else mult
                                        mult = 6.0  if "resnet34" in self.model else mult
                                        mult = 6.0  if "resnet50" in self.model else mult

                                    criteria_for_layer /= mult

            elif self.method == 23:
                # Taylor pruning on gate with computing full gradient
                criteria_for_layer = (self.parameters[layer].full_grad.t()).data.pow(2).view(nunits,-1).sum(dim=1)

            elif self.method == 30:
                # batch normalization based pruning
                # by scale (weight) of the batchnorm
                criteria_for_layer = (self.parameters[layer]).data.abs().view(nunits, -1).sum(dim=1)

            elif self.method == 31:
                # Taylor FO on BN
                if hasattr(self.parameters[layer], "bias"):
                    criteria_for_layer = (self.parameters[layer]*self.parameters[layer].grad +
                                          self.parameters[layer].bias*self.parameters[layer].bias.grad ).data.pow(2).view(nunits,-1).sum(dim=1)
                else:
                    criteria_for_layer = (
                                self.parameters[layer] * self.parameters[layer].grad).data.pow(2).view(nunits, -1).sum(dim=1)

            elif self.method == 40:
                # ORACLE on the fly that reevaluates itslef every pruning step
                criteria_for_layer = np.asarray(self.oracle_dict["loss_list"][layer]).copy()
                self.oracle_dict["loss_list"][layer] = list()
            elif self.method == 50:
                # combinatorial pruning - evaluates all possibilities of removing N neurons
                criteria_for_layer = np.asarray(self.oracle_dict["loss_list"][layer]).copy()
                self.oracle_dict["loss_list"][layer] = list()
            else:
                pass

            if self.iterations_done == 0:
                self.prune_network_accomulate["by_layer"][layer] = criteria_for_layer
            else:
                self.prune_network_accomulate["by_layer"][layer] += criteria_for_layer

        self.iterations_done += 1

    @staticmethod
    def group_criteria(list_criteria_per_layer, group_size=1):
        '''
        Function combine criteria per neuron into groups of size group_size.
        Output is a list of groups organized by layers. Length of output is a number of layers.
        The criterion for the group is computed as an average of member's criteria.
        Input:
        list_criteria_per_layer - list of criteria per neuron organized per layer
        group_size - number of neurons per group

        Output:
        groups - groups organized per layer. Each group element is a tuple of 2: (index of neurons, criterion)
        '''
        groups = list()

        for layer in list_criteria_per_layer:
            layer_groups = list()
            indeces = np.argsort(layer)
            for group_id in range(int(np.ceil(len(layer)/group_size))):
                current_group = slice(group_id*group_size, min((group_id+1)*group_size, len(layer)))
                values = [layer[ind] for ind in indeces[current_group]]
                group = [indeces[current_group], sum(values)]

                layer_groups.append(group)
            groups.append(layer_groups)

        return groups

    def compute_saliency(self):
        '''
        Method performs pruning based on precomputed criteria values. Needs to run after add_criteria()
        '''
        def write_to_debug(what_write_name, what_write_value):
            # Aux function to store information in the text file
            with open(self.log_debug, 'a') as f:
                f.write("{} {}\n".format(what_write_name,what_write_value))

        def nothing(what_write_name, what_write_value):
            pass

        if self.method == 50:
            write_to_debug = nothing

        # compute loss since the last pruning and decide if to prune:
        if self.util_loss_tracker_num > 0:
            validation_error = self.util_loss_tracker / self.util_loss_tracker_num
            validation_error_long = validation_error
            acc = self.util_acc_tracker / self.util_loss_tracker_num
        else:
            print("compute loss and run self.util_add_loss(loss.item()) before running this")
            validation_error = 0.0
            acc = 0.0

        self.util_training_loss = validation_error
        self.util_training_acc = acc

        # reset training loss tracker
        self.util_loss_tracker = 0.0
        self.util_acc_tracker = 0.0
        self.util_loss_tracker_num = 0

        if validation_error > self.pruning_threshold:
            ## if error is big then skip pruning
            print("skipping pruning", validation_error, "(%f)"%validation_error_long, self.pruning_threshold)
            if self.method != 4:
                self.res_pruning = -1
                return -1

        if self.maximum_pruning_iterations <= self.pruning_iterations_done:
            # if reached max number of pruning iterations -> exit
            self.res_pruning = -1
            return -1

        self.full_list_of_criteria = list()

        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            if self.iterations_done > 0:
                # momentum turned to be useless and even reduces performance
                contribution = self.prune_network_accomulate["by_layer"][layer] / self.iterations_done
                if self.pruning_iterations_done == 0 or not self.use_momentum or (self.method in [4, 40, 50]):
                    self.prune_network_accomulate["averaged"][layer] = contribution
                else:
                    # use momentum to accumulate criteria over several pruning iterations:
                    self.prune_network_accomulate["averaged"][layer] = self.momentum_coeff*self.prune_network_accomulate["averaged"][layer]+(1.0- self.momentum_coeff)*contribution

                current_layer = self.prune_network_accomulate["averaged"][layer]
                if not (self.method in [1, 4, 40, 15, 50]):
                    current_layer = current_layer.cpu().numpy()

                if self.l2_normalization_per_layer:
                    eps = 1e-8
                    current_layer = current_layer / (np.linalg.norm(current_layer) + eps)

                self.prune_network_accomulate["averaged_cpu"][layer] = current_layer
            else:
                print("First do some add_criteria iterations")
                exit()

            for unit in range(len(self.parameters[layer])):
                criterion_now = current_layer[unit]

                # make sure that pruned neurons have 0 criteria
                self.prune_network_criteria[layer][unit] =  criterion_now * self.pruning_gates[layer][unit]

                if self.method == 50:
                    self.prune_network_criteria[layer][unit] =  criterion_now

        # count number of neurons
        all_neuron_units, neuron_units = self._count_number_of_neurons()
        self.neuron_units = neuron_units
        self.all_neuron_units = all_neuron_units

        # store criteria_result into file
        if not self.pruning_silent:
            import pickle
            store_criteria = self.prune_network_accomulate["averaged_cpu"]
            pickle.dump(store_criteria, open(self.folder_to_write_debug + "criteria_%04d.pickle"%self.pruning_iterations_done, "wb"))
            if self.pruning_iterations_done == 0:
                pickle.dump(store_criteria, open(self.log_folder + "criteria_%d.pickle"%self.method, "wb"))
            pickle.dump(store_criteria, open(self.log_folder + "criteria_%d_final.pickle"%self.method, "wb"))

        if not self.fixed_criteria:
            self.iterations_done = 0

        # create groups per layer
        groups = self.group_criteria(self.prune_network_criteria, group_size=self.group_size)

        # apply flops regularization
        # if self.flops_regularization > 0.0:
        #     self.apply_flops_regularization(groups, mu=self.flops_regularization)

        # get an array of all criteria from groups
        all_criteria = np.asarray([group[1] for layer in groups for group in layer]).reshape(-1)

        prune_neurons_now = (self.pruned_neurons + self.prune_per_iteration)//self.group_size - 1
        if self.prune_neurons_max != -1:
            prune_neurons_now = min(len(all_criteria)-1, min(prune_neurons_now, self.prune_neurons_max//self.group_size - 1))

        # adaptively estimate threshold given a number of neurons to be removed
        threshold_now = np.sort(all_criteria)[prune_neurons_now]

        if self.method == 50:
            # combinatorial approach
            threshold_now = 0.5
            self.pruning_iterations_done = self.combination_ID
            self.data_logger["combination_ID"].append(self.combination_ID-1)
            self.combination_ID += 1
            self.reset_oracle_pruning()
            print("full_combinatorial: combination ", self.combination_ID)

        self.pruning_iterations_done += 1

        self.log_debug = self.folder_to_write_debug + 'debugOutput_pruning_%08d' % (
            self.pruning_iterations_done) + '.txt'
        write_to_debug("method", self.method)
        write_to_debug("pruned_neurons", self.pruned_neurons)
        write_to_debug("pruning_iterations_done", self.pruning_iterations_done)
        write_to_debug("neuron_units", neuron_units)
        write_to_debug("all_neuron_units", all_neuron_units)
        write_to_debug("threshold_now", threshold_now)
        write_to_debug("groups_total", sum([len(layer) for layer in groups]))

        if self.pruning_iterations_done < self.start_pruning_after_n_iterations:
            self.res_pruning = -1
            return -1

        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            write_to_debug("\nLayer:", layer)
            write_to_debug("units:", len(self.parameters[layer]))

            if self.prune_per_iteration == 0:
                continue

            for group in groups[layer]:
                if group[1] <= threshold_now:
                    for unit in group[0]:
                        # do actual pruning
                        self.pruning_gates[layer][unit] *= 0.0
                        self.parameters[layer].data[unit] *= 0.0

            write_to_debug("pruned_perc:", [np.nonzero(1.0-self.pruning_gates[layer])[0].size, len(self.parameters[layer])])

        # count number of neurons
        all_neuron_units, neuron_units = self._count_number_of_neurons()

        self.pruned_neurons = all_neuron_units-neuron_units

        if self.method == 25:
            self.method_25_first_done = True

        self.threshold_now = threshold_now
        try:
            self.min_criteria_value = (all_criteria[all_criteria > 0.0]).min()
            self.max_criteria_value = (all_criteria[all_criteria > 0.0]).max()
            self.median_criteria_value = np.median(all_criteria[all_criteria > 0.0])
        except:
            self.min_criteria_value = 0.0
            self.max_criteria_value = 0.0
            self.median_criteria_value = 0.0

        # set result to successful
        self.res_pruning = 1

    def _count_number_of_neurons(self):
        '''
        Function computes number of total neurons and number of active neurons
        :return:
        all_neuron_units - number of neurons considered for pruning
        neuron_units     - number of not pruned neurons in the model
        '''
        all_neuron_units = 0
        neuron_units = 0
        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            all_neuron_units += len( self.parameters[layer] )
            for unit in range(len( self.parameters[layer] )):
                if len(self.parameters[layer].data.size()) > 1:
                    statistics = self.parameters[layer].data[unit].abs().sum()
                else:
                    statistics = self.parameters[layer].data[unit]

                if statistics > 0.0:
                    neuron_units += 1

        return all_neuron_units, neuron_units

    def set_weights_oracle_pruning(self):
        '''
        sets gates/weights to zero to evaluate pruning
        will reuse weights for pruning
        only for oracle pruning
        '''

        for layer,if_prune in enumerate(self.prune_layers_oracle):
            if not if_prune:
                continue

            if self.method == 40:
                self.parameters[layer].data = deepcopy(torch.from_numpy(self.stored_weights).cuda())

            for unit in range(len(self.parameters[layer])):
                if self.method == 40:
                    self.pruning_gates[layer][unit] = 1.0

                    if unit == self.oracle_unit:
                        self.pruning_gates[layer][unit] *= 0.0
                        self.parameters[layer].data[unit] *= 0.0

                        # if 'momentum_buffer' in optimizer.state[self.parameters[layer]].keys():
                        #     optimizer.state[self.parameters[layer]]['momentum_buffer'][unit] *= 0.0
        return 1

    def reset_oracle_pruning(self):
        '''
        Method restores weights to original after masking for Oracle pruning
        :return:
        '''
        for layer, if_prune in enumerate(self.prune_layers_oracle):
            if not if_prune:
                continue

            if self.method == 40 or self.method == 50:
                self.parameters[layer].data = deepcopy(torch.from_numpy(self.stored_weights).cuda())

            for unit in range(len( self.parameters[layer])):
                if self.method == 40 or self.method == 50:
                    self.pruning_gates[layer][unit] = 1.0

    def enforce_pruning(self):
        '''
        Method sets parameters ang gates to 0 for pruned neurons.
        Helpful if optimizer will change weights from being zero (due to regularization etc.)
        '''
        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            for unit in range(len(self.parameters[layer])):
                if self.pruning_gates[layer][unit] == 0.0:
                    self.parameters[layer].data[unit] *= 0.0

    def compute_hessian(self, loss):
        '''
        Computes Hessian per layer of the loss with respect to self.parameters, currently implemented only for gates
        '''

        if self.maximum_pruning_iterations <= self.pruning_iterations_done:
            # if reached max number of pruning iterations -> exit
            self.res_pruning = -1
            return -1

        self.temp_hessian = list()
        for layer_indx, parameter in enumerate(self.parameters):
            # print("Computing Hessian current/total layers:",layer_indx,"/",len(self.parameters))
            if self.prune_layers[layer_indx]:
                grad_params = torch.autograd.grad(loss, parameter, create_graph=True)
                length_grad = len(grad_params[0])
                hessian = torch.zeros(length_grad, length_grad)

                cnt = 0
                for parameter_loc in range(len(parameter)):
                    if parameter[parameter_loc].data.cpu().numpy().sum() == 0.0:
                        continue

                    grad_params2 = torch.autograd.grad(grad_params[0][parameter_loc], parameter, create_graph=True)
                    hessian[parameter_loc, :] = grad_params2[0].data

            else:
                length_grad = len(parameter)
                hessian = torch.zeros(length_grad, length_grad)

            self.temp_hessian.append(torch.FloatTensor(hessian.cpu().numpy()).cuda())

    def run_full_oracle(self, model, data, target, criterion, initial_loss):
        '''
        Runs oracle on all data by setting to 0 every neuron and running forward pass
        '''

        # stop adding data if needed
        if self.maximum_pruning_iterations <= self.pruning_iterations_done:
            # if reached max number of pruning iterations -> exit
            self.res_pruning = -1
            return -1

        if self.method == 40:
            # for oracle let's try to do the best possible oracle by evaluating all neurons for each batch
            self.oracle_dict["initial_loss"] += initial_loss
            self.oracle_dict["iterations"]   += 1

            # import pdb; pdb.set_trace()
            if hasattr(self, 'stored_pruning'):
                if self.stored_pruning['use_now']:
                    # load first list of criteria
                    print("use previous computed priors")
                    for layer_index, layer_parameters in enumerate(self.parameters):

                        # start list of estiamtes for the layer if it is empty
                        if len(self.oracle_dict["loss_list"]) < layer_index + 1:
                            self.oracle_dict["loss_list"].append(list())

                        if self.prune_layers[layer_index] == False:
                            continue

                        self.oracle_dict["loss_list"][layer_index] = self.stored_pruning['criteria_start'][layer_index]
                    self.pruned_neurons = self.stored_pruning['neuron_start']
                    return 1

            # do first pass with precomputed values
            for layer_index, layer_parameters in enumerate(self.parameters):
                # start list of estimates for the layer if it is empty
                if len(self.oracle_dict["loss_list"]) < layer_index + 1:
                    self.oracle_dict["loss_list"].append(list())

                if not self.prune_layers[layer_index]:
                    continue
                # copy original prune_layer variable that sets layers to be prunned
                self.prune_layers_oracle = [False, ]*len(self.parameters)
                self.prune_layers_oracle[layer_index] = True
                # store weights for future to recover
                self.stored_weights = deepcopy(self.parameters[layer_index].data.cpu().numpy())

                for neurion_id, neuron in enumerate(layer_parameters):
                    # set neuron to zero
                    self.oracle_unit = neurion_id
                    self.set_weights_oracle_pruning()

                    if self.stored_weights[neurion_id].sum() == 0.0:
                        new_loss = initial_loss
                    else:
                        outputs = model(data)
                        loss = criterion(outputs, target)
                        new_loss = loss.item()

                    # define loss
                    oracle_value = abs(initial_loss - new_loss)
                    # relative loss for testing:
                    # oracle_value = initial_loss - new_loss

                    if len(self.oracle_dict["loss_list"][layer_index]) == 0:
                        self.oracle_dict["loss_list"][layer_index] = [oracle_value, ]
                    elif len(self.oracle_dict["loss_list"][layer_index]) < neurion_id+1:
                        self.oracle_dict["loss_list"][layer_index].append(oracle_value)
                    else:
                        self.oracle_dict["loss_list"][layer_index][neurion_id] += oracle_value

                self.reset_oracle_pruning()

        elif self.method == 50:
            if self.pruning_iterations_done == 0:
                # store weights again
                self.stored_weights = deepcopy(self.parameters[self.fixed_layer].data.cpu().numpy())

            self.set_next_combination()

        else:
            pass
            # print("Full oracle only works with the methods: {}".format(40))

    def report_loss_neuron(self, training_loss, training_acc, train_writer = None, neurons_left = 0):
        '''
        method to store stistics during pruning to the log file
        :param training_loss:
        :param training_acc:
        :param train_writer:
        :param neurons_left:
        :return:
        void
        '''
        if train_writer is not None:
            train_writer.add_scalar('loss_neuron', training_loss, self.all_neuron_units-self.neuron_units)

        self.data_logger["pruning_neurons"].append(self.all_neuron_units-self.neuron_units)
        self.data_logger["pruning_loss"].append(training_loss)
        self.data_logger["pruning_accuracy"].append(training_acc)

        self.write_log_file()

    def write_log_file(self):
        with open(self.data_logger["filename"], "wb") as f:
            pickle.dump(self.data_logger, f)

    def load_mask(self):
        '''Method loads precomputed criteria for pruning
        :return:
        '''
        if not len(self.pruning_mask_from)>0:
            print("pruning_engine.load_mask(): did not find mask file, will load nothing")
        else:
            if not os.path.isfile(self.pruning_mask_from):
                print("pruning_engine.load_mask(): file doesn't exist", self.pruning_mask_from)
                print("pruning_engine.load_mask(): check it, exit,", self.pruning_mask_from)
                exit()

            with open(self.pruning_mask_from, 'rb') as f:
                self.loaded_mask_criteria = pickle.load(f)

            print("pruning_engine.load_mask(): loaded criteria from", self.pruning_mask_from)

    def set_next_combination(self):
        '''
        For combinatorial pruning only
        '''
        if self.method == 50:

            self.oracle_dict["iterations"]   += 1

            for layer_index, layer_parameters in enumerate(self.parameters):

                ##start list of estiamtes for the layer if it is empty
                if len(self.oracle_dict["loss_list"]) < layer_index + 1:
                    self.oracle_dict["loss_list"].append(list())

                if self.prune_layers[layer_index] == False:
                    continue

                nunits = len(layer_parameters)

                comb_num = -1
                found_combination = False
                for it in itertools.combinations(range(nunits), self.starting_neuron):
                    comb_num += 1
                    if comb_num == int(self.combination_ID):
                        found_combination = True
                        break

                # import pdb; pdb.set_trace()
                if not found_combination:
                    print("didn't find needed combination, exit")
                    exit()

                self.prune_layers_oracle = self.prune_layers.copy()
                self.prune_layers_oracle = [False,]*len(self.parameters)
                self.prune_layers_oracle[layer_index] = True

                criteria_for_layer = np.ones((nunits,))
                criteria_for_layer[list(it)] = 0.0

                if len(self.oracle_dict["loss_list"][layer_index]) == 0:
                    self.oracle_dict["loss_list"][layer_index] = criteria_for_layer
                else:
                    self.oracle_dict["loss_list"][layer_index] += criteria_for_layer

    def report_to_tensorboard(self, train_writer, processed_batches):
        '''
        Log data with tensorboard
        '''
        gradient_norm_final_before = self.gradient_norm_final
        train_writer.add_scalar('Neurons_left', self.neuron_units, processed_batches)
        train_writer.add_scalar('Criteria_min', self.min_criteria_value, self.pruning_iterations_done)
        train_writer.add_scalar('Criteria_max', self.max_criteria_value, self.pruning_iterations_done)
        train_writer.add_scalar('Criteria_median', self.median_criteria_value, self.pruning_iterations_done)
        train_writer.add_scalar('Gradient_norm_before', gradient_norm_final_before, self.pruning_iterations_done)
        train_writer.add_scalar('Pruning_threshold', self.threshold_now, self.pruning_iterations_done)

    def util_add_loss(self, training_loss_current, training_acc):
        # keeps track of current loss
        self.util_loss_tracker += training_loss_current
        self.util_acc_tracker  += training_acc
        self.util_loss_tracker_num += 1
        self.loss_tracker_exp.update(training_loss_current)
        # self.acc_tracker_exp.update(training_acc)

    def do_step(self, loss=None, optimizer=None, neurons_left=0, training_acc=0.0):
        '''
        do one step of pruning,
        1) Add importance estimate
        2) checks if loss is above threshold
        3) performs one step of pruning if needed
        '''
        self.iter_step += 1
        niter = self.iter_step

        # # sets pruned weights to zero
        # self.enforce_pruning()

        # stop if pruned maximum amount
        if self.maximum_pruning_iterations <= self.pruning_iterations_done:
            # exit if we pruned enough
            self.res_pruning = -1
            return -1

        # sets pruned weights to zero
        self.enforce_pruning()

        # compute criteria for given batch
        self.add_criteria()

        # small script to keep track of training loss since the last pruning
        self.util_add_loss(loss, training_acc)

        if ((niter-1) % self.frequency == 0) and (niter != 0) and (self.res_pruning==1):
            self.report_loss_neuron(self.util_training_loss, training_acc=self.util_training_acc, train_writer=self.train_writer, neurons_left=neurons_left)

        if niter % self.frequency == 0 and niter != 0:
            # do actual pruning, output: 1 - good, 0 - no pruning

            self.compute_saliency()
            self.set_momentum_zero_sgd(optimizer=optimizer)

            training_loss = self.util_training_loss
            if self.res_pruning == 1:
                print("Pruning: Units", self.neuron_units, "/", self.all_neuron_units, "loss", training_loss, "Zeroed", self.pruned_neurons, "criteria min:{}/max:{:2.7f}".format(self.min_criteria_value,self.max_criteria_value))

    def set_momentum_zero_sgd(self, optimizer=None):
        '''
        Method sets momentum buffer to zero for pruned neurons. Supports SGD only.
        :return:
        void
        '''
        for layer in range(len(self.pruning_gates)):
            if not self.prune_layers[layer]:
                continue
            for unit in range(len(self.pruning_gates[layer])):
                if not self.pruning_gates[layer][unit]:
                    continue
                if 'momentum_buffer' in optimizer.state[self.parameters[layer]].keys():
                    optimizer.state[self.parameters[layer]]['momentum_buffer'][unit] *= 0.0

    def connect_tensorboard(self, tensorboard):
        '''
        Function connects tensorboard to pruning engine
        '''
        self.tensorboard = True
        self.train_writer = tensorboard

    def update_flops(self, stats=None):
        '''
        Function updates flops for potential regularization
        :param stats: a list of flops per parameter
        :return:
        '''
        self.per_layer_flops = list()
        if len(stats["flops"]) < 1:
            return -1
        for pruning_param in self.gates_to_params:
            if isinstance(pruning_param, list):
                # parameter spans many blocks, will aggregate over them
                self.per_layer_flops.append(sum([stats['flops'][a] for a in pruning_param]))
            else:
                self.per_layer_flops.append(stats['flops'][pruning_param])

    def apply_flops_regularization(self, groups, mu=0.1):
        '''
        Function applieregularisation to computed importance per layer
        :param groups: a list of groups organized per layer
        :param mu: regularization coefficient
        :return:
        '''
        if len(self.per_layer_flops) < 1:
            return -1

        for layer_id, layer in enumerate(groups):
            for group in layer:
                # import pdb; pdb.set_trace()
                total_neurons = len(group[0])
                group[1] = group[1] - mu*(self.per_layer_flops[layer_id]*total_neurons)


def prepare_pruning_list(pruning_settings, model, model_name, pruning_mask_from='', name=''):
    '''
    Function returns a list of parameters from model to be considered for pruning.
    Depending on the pruning method and strategy different parameters are selected (conv kernels, BN parameters etc)
    :param pruning_settings:
    :param model:
    :return:
    '''
    # Function creates a list of layer that will be pruned based o user selection

    ADD_BY_GATES = True  # gates add artificially they have weight == 1 and not trained, but gradient is important. see models/lenet.py
    ADD_BY_WEIGHTS = ADD_BY_BN = False

    pruning_method = pruning_settings['method']

    pruning_parameters_list = list()
    if ADD_BY_GATES:

        first_step = True
        prev_module = None
        prev_module2 = None
        print("network structure")
        for module_indx, m in enumerate(model.modules()):
            # print(module_indx, m)
            if hasattr(m, "do_not_update"):
                m_to_add = m

                if (pruning_method != 23) and (pruning_method != 6):
                    for_pruning = {"parameter": m_to_add.weight, "layer": m_to_add,
                                   "compute_criteria_from": m_to_add.weight}
                else:
                    def just_hook(self, grad_input, grad_output):
                        # getting full gradient for parameters
                        # normal backward will provide only averaged gradient per batch
                        # requires to store output of the layer
                        if len(grad_output[0].shape) == 4:
                            self.weight.full_grad = (grad_output[0] * self.output).sum(-1).sum(-1)
                        else:
                            self.weight.full_grad = (grad_output[0] * self.output)

                    if pruning_method == 6:
                        # implement ICLR2017 paper
                        def just_hook(self, grad_input, grad_output):
                            if len(grad_output[0].shape) == 4:
                                self.weight.full_grad_iclr2017 = (grad_output[0] * self.output).abs().mean(-1).mean(
                                    -1).mean(0)
                            else:
                                self.weight.full_grad_iclr2017 = (grad_output[0] * self.output).abs().mean(0)

                    def forward_hook(self, input, output):
                        self.output = output

                    if not len(pruning_mask_from) > 0:
                        # in case mask is precomputed we remove hooks
                        m_to_add.register_forward_hook(forward_hook)
                        m_to_add.register_backward_hook(just_hook)

                    for_pruning = {"parameter": m_to_add.weight, "layer": m_to_add,
                                   "compute_criteria_from": m_to_add.weight}

                if pruning_method in [30, 31]:
                    # for densenets.
                    # add previous layer's value for batch norm pruning

                    if isinstance(prev_module, nn.BatchNorm2d):
                        m_to_add = prev_module
                        print(m_to_add, "yes")
                    else:
                        print(m_to_add, "no")

                    for_pruning = {"parameter": m_to_add.weight, "layer": m_to_add,
                                   "compute_criteria_from": m_to_add.weight}

                if pruning_method in [24, ]:
                    # add previous layer's value for batch norm pruning

                    if isinstance(prev_module, nn.Conv2d):
                        m_to_add = prev_module

                    for_pruning = {"parameter": m_to_add.weight, "layer": m_to_add,
                                   "compute_criteria_from": m_to_add.weight}

                if pruning_method in [0, 2, 3]:
                    # add previous layer's value for batch norm pruning

                    if isinstance(prev_module2, nn.Conv2d):
                        print(module_indx, prev_module2, "yes")
                        m_to_add = prev_module2
                    elif isinstance(prev_module2, nn.Linear):
                        print(module_indx, prev_module2, "yes")
                        m_to_add = prev_module2
                    elif isinstance(prev_module, nn.Conv2d):
                        print(module_indx, prev_module, "yes")
                        m_to_add = prev_module
                    else:
                        print(module_indx, m, "no")

                    for_pruning = {"parameter": m_to_add.weight, "layer": m_to_add,
                                   "compute_criteria_from": m_to_add.weight}

                pruning_parameters_list.append(for_pruning)
            prev_module2 = prev_module
            prev_module = m

    if model_name == "resnet20":
        # prune only even layers as in Rethinking min norm pruning
        pruning_parameters_list = [d for di, d in enumerate(pruning_parameters_list) if (di % 2 == 1 and di > 0)]

    if ("prune_only_skip_connections" in name) and 1:
        # will prune only skip connections (gates around them). Works with ResNets only
        pruning_parameters_list = pruning_parameters_list[:4]

    return pruning_parameters_list

class ExpMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, mom = 0.9):
        self.reset()
        self.mom = mom

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.exp_avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.mean_avg = self.sum / self.count
        self.exp_avg = self.mom*self.exp_avg + (1.0 - self.mom)*self.val
        if self.count == 1:
            self.exp_avg = self.val

if __name__ == '__main__':
    pass
