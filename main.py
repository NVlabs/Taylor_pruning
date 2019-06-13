"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from __future__ import print_function
import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn

# ++++++for pruning
import os, sys
import time
from utils.utils import save_checkpoint, adjust_learning_rate, AverageMeter, accuracy, load_model_pytorch, dynamic_network_change_local, get_conv_sizes, connect_gates_with_parameters_for_flops
from tensorboardX import SummaryWriter

from logger import Logger
from models.lenet import LeNet
from models.vgg_bn import slimmingvgg as vgg11_bn
from models.preact_resnet import *
from pruning_engine import pytorch_pruning, PruningConfigReader, prepare_pruning_list

from utils.group_lasso_optimizer import group_lasso_decay

import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel

import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

import numpy as np
#++++++++end

# code is based on Pytorch example for imagenet
# https://github.com/pytorch/examples/tree/master/imagenet


def str2bool(v):
    # from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/43357954#43357954
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train(args, model, device, train_loader, optimizer, epoch, criterion, train_writer=None, pruning_engine=None):
    """Train for one epoch on the training set also performs pruning"""
    global global_iteration
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_tracker = 0.0
    acc_tracker = 0.0
    loss_tracker_num = 0
    res_pruning = 0

    model.train()
    if args.fixed_network:
        # if network is fixed then we put it to eval mode
        model.eval()

    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # make sure that all gradients are zero
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

        output = model(data)
        loss = criterion(output, target)

        if args.pruning:
            # useful for method 40 and 50 that calculate oracle
            pruning_engine.run_full_oracle(model, data, target, criterion, initial_loss=loss.item())

        # measure accuracy and record loss
        losses.update(loss.item(), data.size(0))

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        top1.update(prec1.item(), data.size(0))
        top5.update(prec5.item(), data.size(0))
        acc_tracker += prec1.item()

        loss_tracker += loss.item()

        loss_tracker_num += 1

        if args.pruning:
            if pruning_engine.needs_hessian:
                pruning_engine.compute_hessian(loss)

        if not (args.pruning and args.pruning_method == 50):
            group_wd_optimizer.step()


        loss.backward()


        # add gradient clipping
        if not args.no_grad_clip:
            # found it useless for our experiments
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step_after will calculate flops and number of parameters left
        # needs to be launched before the main optimizer,
        # otherwise weight decay will make numbers not correct
        if not (args.pruning and args.pruning_method == 50):
            if batch_idx % args.log_interval == 0:
                group_wd_optimizer.step_after()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        global_iteration = global_iteration + 1

        if batch_idx % args.log_interval == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, batch_idx, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1, top5=top5))

            if train_writer is not None:
                train_writer.add_scalar('train_loss_ave', losses.avg, global_iteration)

        if args.pruning:
            # pruning_engine.update_flops(stats=group_wd_optimizer.per_layer_per_neuron_stats)
            pruning_engine.do_step(loss=loss.item(), optimizer=optimizer)

            if args.model == "resnet20" or args.model == "resnet101" or args.dataset == "Imagenet":
                if (pruning_engine.maximum_pruning_iterations == pruning_engine.pruning_iterations_done) and pruning_engine.set_moment_zero:
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            if p.grad is None:
                                continue
                            param_state = optimizer.state[p]
                            if 'momentum_buffer' in param_state:
                                del param_state['momentum_buffer']

                    pruning_engine.set_moment_zero = False

        # if not (args.pruning and args.pruning_method == 50):
        #     if batch_idx % args.log_interval == 0:
        #         group_wd_optimizer.step_after()

        if args.tensorboard and (batch_idx % args.log_interval == 0):

            neurons_left = int(group_wd_optimizer.get_number_neurons(print_output=args.get_flops))
            flops = int(group_wd_optimizer.get_number_flops(print_output=args.get_flops))

            train_writer.add_scalar('neurons_optimizer_left', neurons_left, global_iteration)
            train_writer.add_scalar('neurons_optimizer_flops_left', flops, global_iteration)
        else:
            if args.get_flops:
                neurons_left = int(group_wd_optimizer.get_number_neurons(print_output=args.get_flops))
                flops = int(group_wd_optimizer.get_number_flops(print_output=args.get_flops))

        if args.limit_training_batches != -1:
            if args.limit_training_batches < batch_idx:
                # return from training step, unsafe and was not tested correctly
                print("return from training step, unsafe and was not tested correctly")
                return 0

    # print number of parameters left:
    if args.tensorboard:
        print('neurons_optimizer_left', neurons_left, global_iteration)


def validate(args, test_loader, model, device, criterion, epoch, train_writer=None):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for data_test in test_loader:
            data, target = data_test

            data = data.to(device)

            output = model(data)

            if args.get_inference_time:
                iterations_get_inference_time = 100
                start_get_inference_time = time.time()
                for it in range(iterations_get_inference_time):
                    output = model(data)
                end_get_inference_time = time.time()
                print("time taken for %d iterations, per-iteration is: "%(iterations_get_inference_time), (end_get_inference_time - start_get_inference_time)*1000.0/float(iterations_get_inference_time), "ms")

            target = target.to(device)
            loss = criterion(output, target)

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print(' * Prec@1 {top1.avg:.3f}, Prec@5 {top5.avg:.3f}, Time {batch_time.sum:.5f}, Loss: {losses.avg:.3f}'.format(top1=top1, top5=top5,batch_time=batch_time, losses = losses) )
    # log to TensorBoard
    if train_writer is not None:
        train_writer.add_scalar('val_loss', losses.avg, epoch)
        train_writer.add_scalar('val_acc', top1.avg, epoch)

    return top1.avg, losses.avg



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--world_size', type=int, default=1,
                        help='number of GPUs to use')

    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--lr-decay-every', type=int, default=100,
                        help='learning rate decay by 10 every X epochs')
    parser.add_argument('--lr-decay-scalar', type=float, default=0.1,
                        help='--')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--run_test', default=False,  type=str2bool, nargs='?',
                        help='run test only')

    parser.add_argument('--limit_training_batches', type=int, default=-1,
                        help='how many batches to do per training, -1 means as many as possible')

    parser.add_argument('--no_grad_clip', default=False,  type=str2bool, nargs='?',
                        help='turn off gradient clipping')

    parser.add_argument('--get_flops', default=False,  type=str2bool, nargs='?',
                        help='add hooks to compute flops')

    parser.add_argument('--get_inference_time', default=False,  type=str2bool, nargs='?',
                        help='runs valid multiple times and reports the result')

    parser.add_argument('--mgpu', default=False,  type=str2bool, nargs='?',
                        help='use data paralization via multiple GPUs')

    parser.add_argument('--dataset', default="MNIST", type=str,
                        help='dataset for experiment, choice: MNIST, CIFAR10', choices= ["MNIST", "CIFAR10", "Imagenet"])

    parser.add_argument('--data', metavar='DIR', default='/imagenet', help='path to imagenet dataset')

    parser.add_argument('--model', default="lenet3", type=str,
                        help='model selection, choices: lenet3, vgg, mobilenetv2, resnet18',
                        choices=["lenet3", "vgg", "mobilenetv2", "resnet18", "resnet152", "resnet50", "resnet50_noskip",
                                 "resnet20", "resnet34", "resnet101", "resnet101_noskip", "densenet201_imagenet",
                                 'densenet121_imagenet'])

    parser.add_argument('--tensorboard', type=str2bool, nargs='?',
                        help='Log progress to TensorBoard')

    parser.add_argument('--save_models', default=True, type=str2bool, nargs='?',
                        help='if True, models will be saved to the local folder')


    # ============================PRUNING added
    parser.add_argument('--pruning_config', default=None, type=str,
                        help='path to pruning configuration file, will overwrite all pruning parameters in arguments')

    parser.add_argument('--group_wd_coeff', type=float, default=0.0,
                        help='group weight decay')
    parser.add_argument('--name', default='test', type=str,
                        help='experiment name(folder) to store logs')

    parser.add_argument('--augment', default=False, type=str2bool, nargs='?',
                            help='enable or not augmentation of training dataset, only for CIFAR, def False')

    parser.add_argument('--load_model', default='', type=str,
                        help='path to model weights')

    parser.add_argument('--pruning', default=False, type=str2bool, nargs='?',
                        help='enable or not pruning, def False')

    parser.add_argument('--pruning-threshold', '--pt', default=100.0, type=float,
                        help='Max error perc on validation set while pruning (default: 100.0 means always prune)')

    parser.add_argument('--pruning-momentum', default=0.0, type=float,
                        help='Use momentum on criteria between pruning iterations, def 0.0 means no momentum')

    parser.add_argument('--pruning-step', default=15, type=int,
                        help='How often to check loss and do pruning step')

    parser.add_argument('--prune_per_iteration', default=10, type=int,
                        help='How many neurons to remove at each iteration')

    parser.add_argument('--fixed_layer', default=-1, type=int,
                        help='Prune only a given layer with index, use -1 to prune all')

    parser.add_argument('--start_pruning_after_n_iterations', default=0, type=int,
                        help='from which iteration to start pruning')

    parser.add_argument('--maximum_pruning_iterations', default=1e8, type=int,
                        help='maximum pruning iterations')

    parser.add_argument('--starting_neuron', default=0, type=int,
                        help='starting position for oracle pruning')

    parser.add_argument('--prune_neurons_max', default=-1, type=int,
                        help='prune_neurons_max')

    parser.add_argument('--pruning-method', default=0, type=int,
                        help='pruning method to be used, see readme.md')

    parser.add_argument('--pruning_fixed_criteria', default=False, type=str2bool, nargs='?',
                        help='enable or not criteria reevaluation, def False')

    parser.add_argument('--fixed_network', default=False,  type=str2bool, nargs='?',
                        help='fix network for oracle or criteria computation')

    parser.add_argument('--zero_lr_for_epochs', default=-1, type=int,
                        help='Learning rate will be set to 0 for given number of updates')

    parser.add_argument('--dynamic_network', default=False,  type=str2bool, nargs='?',
                        help='Creates a new network graph from pruned model, works with ResNet-101 only')

    parser.add_argument('--use_test_as_train', default=False,  type=str2bool, nargs='?',
                        help='use testing dataset instead of training')

    parser.add_argument('--pruning_mask_from', default='', type=str,
                        help='path to mask file precomputed')

    parser.add_argument('--compute_flops', default=True,  type=str2bool, nargs='?',
                        help='if True, will run dummy inference of batch 1 before training to get conv sizes')



    # ============================END pruning added

    best_prec1 = 0
    global global_iteration
    global group_wd_optimizer
    global_iteration = 0

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    args.distributed = args.world_size > 1
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=0)

    device = torch.device("cuda" if use_cuda else "cpu")

    if args.model == "lenet3":
        model = LeNet(dataset=args.dataset)
    elif args.model == "vgg":
        model = vgg11_bn(pretrained=True)
    elif args.model == "resnet18":
        model = PreActResNet18()
    elif (args.model == "resnet50") or (args.model == "resnet50_noskip"):
        if args.dataset == "CIFAR10":
            model = PreActResNet50(dataset=args.dataset)
        else:
            from models.resnet import resnet50
            skip_gate = True
            if "noskip" in args.model:
                skip_gate = False

            if args.pruning_method not in [22, 40]:
                skip_gate = False
            model = resnet50(skip_gate=skip_gate)
    elif args.model == "resnet34":
        if not (args.dataset == "CIFAR10"):
            from models.resnet import resnet34
            model = resnet34()
    elif "resnet101" in args.model:
        if not (args.dataset == "CIFAR10"):
            from models.resnet import resnet101
            if args.dataset == "Imagenet":
                classes = 1000

            if "noskip" in args.model:
                model = resnet101(num_classes=classes, skip_gate=False)
            else:
                model = resnet101(num_classes=classes)

    elif args.model == "resnet20":
        if args.dataset == "CIFAR10":
            NotImplementedError("resnet20 is not implemented in the current project")
            # from models.resnet_cifar import resnet20
            # model = resnet20()
    elif args.model == "resnet152":
        model = PreActResNet152()
    elif args.model == "densenet201_imagenet":
        from models.densenet_imagenet import DenseNet201
        model = DenseNet201(gate_types=['output_bn'], pretrained=True)
    elif args.model == "densenet121_imagenet":
        from models.densenet_imagenet import DenseNet121
        model = DenseNet121(gate_types=['output_bn'], pretrained=True)
    else:
        print(args.model, "model is not supported")

    # dataset loading section
    if args.dataset == "MNIST":
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    elif args.dataset == "CIFAR10":
        # Data loading code
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])

        if args.augment:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
            ])

        kwargs = {'num_workers': 8, 'pin_memory': True}
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=True, download=True,
                             transform=transform_train),
            batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False, transform=transform_test),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    elif args.dataset == "Imagenet":
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        kwargs = {'num_workers': 16}

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            sampler=train_sampler, pin_memory=True, **kwargs)

        if args.use_test_as_train:
            train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=(train_sampler is None), **kwargs)


        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False, pin_memory=True, **kwargs)

    ####end dataset preparation

    if args.dynamic_network:
        # attempts to load pruned model and modify it be removing pruned channels
        # works for resnet101 only
        if (len(args.load_model) > 0) and (args.dynamic_network):
            if os.path.isfile(args.load_model):
                load_model_pytorch(model, args.load_model, args.model)

            else:
                print("=> no checkpoint found at '{}'".format(args.load_model))
                exit()

        dynamic_network_change_local(model)

        # save the model
        log_save_folder = "%s"%args.name
        if not os.path.exists(log_save_folder):
            os.makedirs(log_save_folder)

        if not os.path.exists("%s/models" % (log_save_folder)):
            os.makedirs("%s/models" % (log_save_folder))

        model_save_path = "%s/models/pruned.weights"%(log_save_folder)
        model_state_dict = model.state_dict()
        if args.save_models:
            save_checkpoint({
                'state_dict': model_state_dict
            }, False, filename = model_save_path)

    print("model is defined")

    # aux function to get size of feature maps
    # First it adds hooks for each conv layer
    # Then runs inference with 1 image
    output_sizes = get_conv_sizes(args, model)

    if use_cuda and not args.mgpu:
        model = model.to(device)
    elif args.distributed:
        model.cuda()
        print("\n\n WARNING: distributed pruning was not verified and might not work correctly")
        model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.mgpu:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.to(device)

    print("model is set to device: use_cuda {}, args.mgpu {}, agrs.distributed {}".format(use_cuda, args.mgpu, args.distributed))

    weight_decay = args.wd
    if args.fixed_network:
        weight_decay = 0.0

    # remove updates from gate layers, because we want them to be 0 or 1 constantly
    if 1:
        parameters_for_update = []
        parameters_for_update_named = []
        for name, m in model.named_parameters():
            if "gate" not in name:
                parameters_for_update.append(m)
                parameters_for_update_named.append((name, m))
            else:
                print("skipping parameter", name, "shape:", m.shape)

    total_size_params = sum([np.prod(par.shape) for par in parameters_for_update])
    print("Total number of parameters, w/o usage of bn consts: ", total_size_params)

    optimizer = optim.SGD(parameters_for_update, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay)

    if 1:
        # helping optimizer to implement group lasso (with very small weight that doesn't affect training)
        # will be used to calculate number of remaining flops and parameters in the network
        group_wd_optimizer = group_lasso_decay(parameters_for_update, group_lasso_weight=args.group_wd_coeff, named_parameters=parameters_for_update_named, output_sizes=output_sizes)

    cudnn.benchmark = True

    # define objective
    criterion = nn.CrossEntropyLoss()

    ###=======================added for pruning
    # logging part
    log_save_folder = "%s"%args.name
    if not os.path.exists(log_save_folder):
        os.makedirs(log_save_folder)

    if not os.path.exists("%s/models" % (log_save_folder)):
        os.makedirs("%s/models" % (log_save_folder))

    train_writer = None
    if args.tensorboard:
        try:
            # tensorboardX v1.6
            train_writer = SummaryWriter(log_dir="%s"%(log_save_folder))
        except:
            # tensorboardX v1.7
            train_writer = SummaryWriter(logdir="%s"%(log_save_folder))

    time_point = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    textfile = "%s/log_%s.txt" % (log_save_folder, time_point)
    stdout = Logger(textfile)
    sys.stdout = stdout
    print(" ".join(sys.argv))

    # initializing parameters for pruning
    # we can add weights of different layers or we can add gates (multiplies output with 1, useful only for gradient computation)
    pruning_engine = None
    if args.pruning:
        pruning_settings = dict()
        if not (args.pruning_config is None):
            pruning_settings_reader = PruningConfigReader()
            pruning_settings_reader.read_config(args.pruning_config)
            pruning_settings = pruning_settings_reader.get_parameters()

        # overwrite parameters from config file with those from command line
        # needs manual entry here
        # user_specified = [key for key in vars(default_args).keys() if not (vars(default_args)[key]==vars(args)[key])]
        # argv_of_interest = ['pruning_threshold', 'pruning-momentum', 'pruning_step', 'prune_per_iteration',
        #                     'fixed_layer', 'start_pruning_after_n_iterations', 'maximum_pruning_iterations',
        #                     'starting_neuron', 'prune_neurons_max', 'pruning_method']

        has_attribute = lambda x: any([x in a for a in sys.argv])

        if has_attribute('pruning-momentum'):
            pruning_settings['pruning_momentum'] = vars(args)['pruning_momentum']
        if has_attribute('pruning-method'):
            pruning_settings['method'] = vars(args)['pruning_method']

        pruning_parameters_list = prepare_pruning_list(pruning_settings, model, model_name=args.model,
                                                       pruning_mask_from=args.pruning_mask_from, name=args.name)
        print("Total pruning layers:", len(pruning_parameters_list))

        folder_to_write = "%s"%log_save_folder+"/"
        log_folder = folder_to_write

        pruning_engine = pytorch_pruning(pruning_parameters_list, pruning_settings=pruning_settings, log_folder=log_folder)

        pruning_engine.connect_tensorboard(train_writer)
        pruning_engine.dataset = args.dataset
        pruning_engine.model = args.model
        pruning_engine.pruning_mask_from = args.pruning_mask_from
        pruning_engine.load_mask()
        gates_to_params = connect_gates_with_parameters_for_flops(args.model, parameters_for_update_named)
        pruning_engine.gates_to_params = gates_to_params

    ###=======================end for pruning
    # loading model file
    if (len(args.load_model) > 0) and (not args.dynamic_network):
        if os.path.isfile(args.load_model):
            load_model_pytorch(model, args.load_model, args.model)
        else:
            print("=> no checkpoint found at '{}'".format(args.load_model))
            exit()

    if args.tensorboard and 0:
        if args.dataset == "CIFAR10":
            dummy_input = torch.rand(1, 3, 32, 32).to(device)
        elif args.dataset == "Imagenet":
            dummy_input = torch.rand(1, 3, 224, 224).to(device)

        train_writer.add_graph(model, dummy_input)

    for epoch in range(1, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(args, optimizer, epoch, args.zero_lr_for_epochs, train_writer)

        if not args.run_test and not args.get_inference_time:
            train(args, model, device, train_loader, optimizer, epoch, criterion, train_writer=train_writer, pruning_engine=pruning_engine)

        if args.pruning:
            # skip validation error calculation and model saving
            if pruning_engine.method == 50: continue

        # evaluate on validation set
        prec1, _ = validate(args, test_loader, model, device, criterion, epoch, train_writer=train_writer)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        model_save_path = "%s/models/checkpoint.weights"%(log_save_folder)
        model_state_dict = model.state_dict()
        if args.save_models:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_state_dict,
                'best_prec1': best_prec1,
            }, is_best, filename=model_save_path)


if __name__ == '__main__':
    main()
