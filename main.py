import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import os
import argparse
import logging
import time
import numpy as np
import random
from collections import defaultdict

from robustbench.data import load_cifar10c, load_cifar100c, load_imagenetc
from robustbench.utils import clean_accuracy as accuracy

from conf import _C as cfg_default
from yacs.config import CfgNode as CN
from functools import partial
from copy import deepcopy

import tta.norm as norm
import tta.tent as tent
import tta.lntent as lntent
import archs
import edgeFilter

logger = logging.getLogger('Edge')
logger.setLevel(logging.INFO)

def print_args(logger, args, cfg=None):
    logger.info("***************")
    logger.info("** Arguments **")
    logger.info("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    arg_string = ""
    for key in optkeys:
        arg_string += ("{}:{},  ".format(key, args.__dict__[key]))
    logger.info(arg_string)
    logger.info(" ")

    if cfg is not None:
        logger.info("************")
        logger.info("** Config **")
        logger.info("************")
        logger.info(f'\n{cfg}\n')

class lossLogger():
    def __init__(self):
        self.data = defaultdict(list)

    def add(self, input_dict):
        for key, value in input_dict.items():
            self.data[key].append(value)

    def reset(self):
        self.data = defaultdict(list)

    def log(self):
        return {key: sum(values) / len(values) for key, values in self.data.items()}

    def __str__(self):
        averages = {key: round(sum(values) / len(values), 4) for key, values in self.data.items()}
        return f'Avg : {averages}'

class Trainer():
    def __init__(self, cfg, model, LAYER_NAMES, trainloader, testloader, criterion, optimizer, device, wandb):
        self.cfg = cfg
        self.lambda_l1 = cfg.OPTIM.LAMBDA_L1
        self.epoch = 0
        self.iter = 0

        self.model = model
        self.is_vit = 'ViT' in cfg.MODEL.ARCH
        self.LAYER_NAMES = LAYER_NAMES
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.wandb = wandb
    
        # Resgister hook functions
        if self.is_vit:
            for idx in range(len(model.encoder.layers)):
                hook = partial(self.vit_hook_fn, layer_name=f"layer{idx}")
                model.encoder.layers[idx].register_forward_hook(hook)
        else:
            for name, layer in model.named_children():
                if name in LAYER_NAMES:  # Assuming layer_names is a list of layer names you're interested in
                    hook = partial(self.hook_fn, layer_name=name)
                    layer.register_forward_hook(hook)

    
    # For logging filter
    def hook_fn(self, module, input, output, layer_name):
        if (self.iter+1) % self.cfg.LOG_PERIOD == 0:
            channel_wise_norm = torch.norm(output, dim=(2, 3)).cpu().detach().numpy()
            # Flatten and calculate norms (shape: (b*c*h*w,))
            # flat_norms = output.flatten(start_dim=1).norm(dim=1).cpu().detach().numpy()
            density = (output > 0).float().mean().cpu().detach().numpy()

            # Logger
            logger.info(f'[{layer_name}] Density: {density:.04f}')

            # Log the histograms to WandB
            if self.wandb:
                # wandb.log({f"{layer_name} Flat Norms": wandb.Histogram(flat_norms)})
                wandb.log({f"{layer_name} Channel-wise Norms": wandb.Histogram(channel_wise_norm)})
                wandb.log({f"{layer_name} Density": density})

    def vit_hook_fn(self, module, input, output, layer_name):
        if (self.iter+1) % self.cfg.LOG_PERIOD == 0:
            channel_wise_norm = torch.norm(output, dim=-1).cpu().detach().numpy()
            # Flatten and calculate norms (shape: (b*c*h*w,))
            # flat_norms = output.flatten(start_dim=1).norm(dim=1).cpu().detach().numpy()
            density = (output > 0).float().mean().cpu().detach().numpy()

            # Logger
            logger.info(f'[{layer_name}] Density: {density:.04f}')

            # Log the histograms to WandB
            if self.wandb:
                wandb.log({f"{layer_name} Channel-wise Norms": wandb.Histogram(channel_wise_norm)})
                wandb.log({f"{layer_name} Density": density})

    def train(self):
        cfg = self.cfg
        model = self.model
        trainloader = self.trainloader
        device = self.device
        optimizer = self.optimizer
        log_period = self.cfg.LOG_PERIOD

        model.train()
        loss_logger = lossLogger()

        # Training loop
        for self.epoch in range(1, cfg.TRAIN.MAX_EPOCH+1):  # loop over the dataset multiple times
            
            for self.iter, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss_x = self.criterion(outputs, labels)

                # Weight regularization
                if cfg.OPTIM.REG:
                    loss_l1 = 0

                    if self.is_vit:
                        for idx, layer in enumerate(model.encoder.layers):
                            if cfg.MODEL.FILTER.POSITION >= idx:
                                loss_l1 += sum(param.abs().sum() for param in layer.parameters())
                    else:
                        for idx, layer_name in enumerate(self.LAYER_NAMES): # ['conv1', 'block1', 'block2', 'block3']
                            if cfg.MODEL.FILTER.POSITION >= idx:
                                loss_l1 += sum(param.abs().sum() for param in model._modules[layer_name].parameters())
                    
                    loss_l1 = self.lambda_l1 * loss_l1

                # Model update
                loss_total = loss_x
                loss_logger.add({'loss_x': loss_x.item()})
                
                if cfg.OPTIM.REG:
                    loss_total += loss_l1
                    loss_logger.add({'loss_l1': loss_l1.item()})

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                if (self.iter+1) % log_period == 0:    # log mini-batches
                    logger.info(f'epoch [{self.epoch}/{cfg.TRAIN.MAX_EPOCH}] iter [{self.iter+1}/{len(trainloader)}] {loss_logger} {model.filter}')
                    if self.wandb: wandb.log(loss_logger.log())
                    loss_logger.reset()

                    if cfg.MODEL.FILTER.MODE != 'conv':
                        if self.wandb: wandb.log(model.filter.log())
                        model.filter.reset()

        logger.info('Training Done\n')

    # Evaluation
    def eval(self):
        model = self.model
        if self.cfg.MODEL.FILTER.MODE != 'conv': model.filter.reset()
        testloader = self.testloader
        device = self.device

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        logger.info(f'Accuracy test data: {(100 * correct / total)} % {model.filter}')
        if self.wandb: wandb.log({"Eval acc": 100 * correct/ total})

        logger.info('Evaluation Done\n')

    def tta(self, adaptation):
        model = deepcopy(self.model)
        cfg = self.cfg

        if adaptation == "none":
            logger.info("test-time adaptation: NONE")
            model.eval()
        if adaptation == "norm":
            if self.is_vit:
                pass        # Cannot use NORM for ViT
            else:
                logger.info("test-time adaptation: NORM")
                model = norm.Norm(model)
        if adaptation == "tent":
            logger.info("test-time adaptation: TENT")
            if self.is_vit:
                model = lntent.configure_model(model)
                params, param_names = lntent.collect_params(model)
                optimizer = getattr(optim, cfg.OPTIM.METHOD)(params, lr=cfg.OPTIM.LR)
                model = lntent.LNTent(model, optimizer)
            else:
                model = tent.configure_model(model)
                params, param_names = tent.collect_params(model)
                optimizer = getattr(optim, cfg.OPTIM.METHOD)(params, lr=cfg.OPTIM.LR)
                model = tent.Tent(model, optimizer)
            
        # evaluate on each severity and type of corruption in turn
        for severity in cfg.TTA.CORRUPTION.SEVERITY:
            accs = []
            for corruption_type in cfg.TTA.CORRUPTION.TYPE:
                # reset adaptation for each combination of corruption x severity
                # note: for evaluation protocol, but not necessarily needed
                if adaptation == 'none':
                    if self.cfg.MODEL.FILTER.MODE != 'conv': model.filter.reset()
                else:
                    model.reset()
                    if self.cfg.MODEL.FILTER.MODE != 'conv': model.model.filter.reset()
                    logger.info("resetting model")

                if self.cfg.DATA.DATASET == 'cifar10':
                    x_test, y_test = load_cifar10c(cfg.TTA.CORRUPTION.NUM_EX, severity, cfg.DATA.ROOT, True, [corruption_type])
                elif self.cfg.DATA.DATASET == 'cifar100':
                    x_test, y_test = load_cifar100c(cfg.TTA.CORRUPTION.NUM_EX, severity, cfg.DATA.ROOT, True, [corruption_type])
                elif self.cfg.DATA.DATASET == 'imageNet':
                    x_test, y_test = load_imagenetc(5000, severity, cfg.DATA.ROOT, True, [corruption_type])
                else:
                    logger.error('Dataset not found')
                    raise NotImplementedError

                if self.is_vit:
                    x_test = F.interpolate(x_test, size=(224, 224), mode='bilinear', align_corners=False)
                x_test, y_test = x_test.cuda(), y_test.cuda()
                acc = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE)
                accs.append(acc)
                err = 1. - acc
                # logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")
                if adaptation == "none":
                    logger.info(f"accuracy % [{corruption_type}{severity}]: {acc:.2%} {model.filter}")
                else:
                    logger.info(f"accuracy % [{corruption_type}{severity}]: {acc:.2%} {model.model.filter}")
            accs = np.array(accs)
            logger.info(f"Mean accuracy [severity {severity}]: {accs.mean():.2%}")
            if self.wandb: wandb.log({"TTA acc": accs.mean()})

        logger.info(f'{adaptation} method TTA Done\n')

def extend_cfg(cfg):
    # Use in case you want to add more configs instantly

    # cfg.NEW_CONFIG = CN()
    # cfg.NEW_CONFIG.NEW_PARAM = 0
    pass

def reset_cfg(cfg, args):
    cfg.MODEL.FILTER.POSITION = args.pos
    cfg.MODEL.ARCH = args.arch
    cfg.MODEL.FILTER.MODE = args.mode
    cfg.SEED = args.seed
    cfg.GPU = args.gpu
    cfg.OPTIM.REG = args.reg
    
    if args.root: cfg.DATA.ROOT = args.root
    if args.dataset: cfg.DATA.DATASET = args.dataset
    if args.lr: cfg.OPTIM.LR = args.lr
    if args.tta: cfg.TTA.ADAPTATION = args.tta
    if args.log_period: cfg.LOG_PERIOD = args.log_period
    if args.bsz: cfg.TRAIN.BATCH_SIZE = args.bsz
    if args.epoch: cfg.TRAIN.MAX_EPOCH = args.epoch


def setup_cfg(args):
    cfg = cfg_default.clone()
    extend_cfg(cfg)
    cfg.merge_from_file(args.config)

    reset_cfg(cfg, args)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    return cfg


def main(args):
    if args.wandb:
        import wandb

    # Output Formatting
    output_time = time.strftime('%Y-%m-%d_%H%M%S', time.localtime())
    config_version = f'{args.config.split("/")[-1].split(".")[0]}'
    output_dir = os.path.join('output', args.dataset, args.arch, args.mode, f'layer_{args.pos}', f'reg_{args.reg}', config_version, f'seed{args.seed}')
    wandb_name = f'{args.dataset}_{args.arch}_{args.mode}_layer{args.pos}_reg_{args.reg}_{config_version}_{output_time}'
    os.makedirs(output_dir, exist_ok=True)    

    if args.wandb: wandb.init(project='Edge Filter', name=wandb_name)


    # Set Configs
    cfg = setup_cfg(args)
    
    # Seed fix
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
    # Logger
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s : %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_{output_time}.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
   
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    print_args(logger, args, cfg)


    # Data loading
    is_vit = 'ViT' in cfg.MODEL.ARCH

    if 'cifar' in cfg.DATA.DATASET:
        if is_vit:
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            transform_test = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
        
        if cfg.DATA.DATASET == 'cifar10':
            num_classes = 10
            trainset = datasets.CIFAR10(root=cfg.DATA.ROOT, train=True, download=True, transform=transform_train)
            trainloader = DataLoader(trainset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=8)
            testset = datasets.CIFAR10(root=cfg.DATA.ROOT, train=False, download=True, transform=transform_test)
            testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
        elif cfg.DATA.DATASET == 'cifar100':
            num_classes = 100
            trainset = datasets.CIFAR100(root=cfg.DATA.ROOT, train=True, download=True, transform=transform_train)
            trainloader = DataLoader(trainset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=8)
            testset = datasets.CIFAR100(root=cfg.DATA.ROOT, train=False, download=True, transform=transform_test)
            testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

    elif 'imageNet' in cfg.DATA.DATASET:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        num_classes=1000
        folder = 'imagenet'
        train_dataset = datasets.ImageNet(root=os.path.join(cfg.DATA.ROOT, folder), split='train', transform=transform_train)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=8)
        test_dataset = datasets.ImageNet(root=os.path.join(cfg.DATA.ROOT, folder), split='val', transform=transform_test)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=8)

    else:
        logger.error('Dataset not found')
        raise NotImplementedError


    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch_list = [name for name in dir(archs) if isinstance(getattr(archs, name), type) and getattr(archs, name).__module__ == archs.__name__]
    arch_list = [model.replace("_", "-") for model in arch_list]
    
    if cfg.MODEL.ARCH in arch_list:
        model_name = cfg.MODEL.ARCH.replace("-", "_")
    else:
        logger.error(f"Arch '{cfg.MODEL.ARCH}' not found. Arch list : {arch_list}")
        raise NotImplementedError

    arch = getattr(archs, model_name)
    if is_vit:
        base_model = arch(num_classes).model.to(device)
    else:
        base_model = arch(num_classes).to(device)

    # Add Filter
    if cfg.MODEL.ARCH in ['WRN-28-10', 'WRN-40-2']:
        model = edgeFilter.EdgeWRN(cfg, base_model).to(device)
        LAYER_NAMES = ['conv1','block1','block2','block3']    
    elif cfg.MODEL.ARCH in ['ViT-B32', 'ViT-B16']:
        model = edgeFilter.EdgeViT(cfg, base_model).to(device)
        LAYER_NAMES = []
    else:
        raise NotImplementedError


    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, cfg.OPTIM.METHOD)(model.parameters(), lr=cfg.OPTIM.LR)
    
    # Build Trainer
    trainer = Trainer(cfg, model, LAYER_NAMES, trainloader, testloader, criterion, optimizer, device, args.wandb)

    if os.path.exists(os.path.join(output_dir, 'ckpt.pt')):
        model.load_state_dict(torch.load(os.path.join(output_dir, 'ckpt.pt')))
        logger.info('Previous Model loaded')
    else:
        trainer.train()

        # Save model
        torch.save(trainer.model.state_dict(), os.path.join(output_dir, 'ckpt.pt'))
        logger.info('Model saved!')

    # Eval
    trainer.eval()
    
    # TTA
    trainer.tta('none')
    if not is_vit: trainer.tta('norm')
    trainer.tta('tent')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Edge Filter Training Arguments')

    #Config
    parser.add_argument('--config', type=str, help='config file')
    parser.add_argument('--dataset_config', type=str, default='', help='dataset config file')

    #Data
    parser.add_argument('--root', type=str, default='/data', help='root path to dataset')
    parser.add_argument('--dataset', type=str, default='cifar10')
    
    #Model
    parser.add_argument('--arch', type=str, default= 'WRN-28-10', help='select architecture')
    parser.add_argument('--mode', type=str, help='select filter')
    parser.add_argument('--pos', type=int, default=0,  help='filter position')

    #Optim
    parser.add_argument('--reg', action='store_true', help='apply l1 regularlization')

    parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--bsz', type=int, default=128, help='training batch size')
    parser.add_argument('--lr', type=float, help='training learning rate')

    # TTA
    parser.add_argument('--tta', type=str, default='none', help='tta type') # none, norm, tent


    #ENV
    parser.add_argument('--log_period', type=int, default=100, help='iters per logging')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0', help='select gpu index to use')
    parser.add_argument('--wandb', action='store_true', help='use wandb logging')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)