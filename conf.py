from yacs.config import CfgNode as CN

_C = CN()

# ----------------------------- Model options ------------------------------- #
_C.MODEL = CN()
_C.MODEL.ARCH = 'WRN-28-10'

_C.MODEL.FILTER = CN()
_C.MODEL.FILTER.MODE = 'average'
_C.MODEL.FILTER.POSITION = 0
_C.MODEL.FILTER.KERNEL = 3


_C.MODEL.TTA = CN()
# Choice of (source, norm, tent)
# - source: baseline without adaptation
# - norm: test-time normalization
# - tent: test-time entropy minimization (ours)
_C.MODEL.TTA.ADAPTATION = 'source'

# By default tent is online, with updates persisting across batches.
# To make adaptation episodic, and reset the model for each batch, choose True.
_C.MODEL.TTA.EPISODIC = False


# ----------------------------- Data options -------------------------------- #
_C.DATA = CN()

# Root directory of dataset
_C.DATA.ROOT = '/data'

# Dataset for evaluation
_C.DATA.DATASET = 'cifar10'


# ----------------------------- Training options ---------------------------- #
_C.TRAIN = CN()

# Number of epochs  
_C.TRAIN.MAX_EPOCH = 50

# Batch size of training
_C.TRAIN.BATCH_SIZE = 128


# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM = CN()

# Learning rate
_C.OPTIM.LR = 1e-3

# Choices: Adam, SGD
_C.OPTIM.METHOD = 'Adam'

# Beta
_C.OPTIM.BETA = 0.9

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# L1 regularization
_C.OPTIM.REG = False

# L1 regularization weight
_C.OPTIM.LAMBDA_L1 = 5e-6


# ------------------------------- Testing options --------------------------- #
_C.TEST = CN()

# Batch size for evaluation (and updates for norm + tent)
_C.TEST.BATCH_SIZE = 32

# ------------------------------- Testing options --------------------------- #
_C.TTA = CN()

# Batch size for evaluation (and updates for norm + tent)
_C.TTA.ADAPTATION = 'none'

_C.TTA.CORRUPTION = CN()

# Check https://github.com/hendrycks/robustness for corruption details
_C.TTA.CORRUPTION.TYPE = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness', 'contrast',
                      'elastic_transform', 'pixelate', 'jpeg_compression']

_C.TTA.CORRUPTION.SEVERITY = [5,]  # [5, 4, 3, 2, 1]

# Number of examples to evaluate (10000 for all samples in CIFAR-10)
_C.TTA.CORRUPTION.NUM_EX = 10000


# ------------------------------- Batch norm options ------------------------ #
_C.BN = CN()

# BN epsilon
_C.BN.EPS = 1e-5

# BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
_C.BN.MOM = 0.1


# ---------------------------------- Misc options --------------------------- #

# Seed Fix
_C.SEED = 1

# GPU
_C.GPU = 0

# Logging period
_C.LOG_PERIOD = 100

# Output directory
_C.SAVE_DIR = "./output"

# Log destination (in SAVE_DIR)
# _C.LOG_DEST = "log.txt"

# Config destination (in SAVE_DIR)
# _C.CFG_DEST = "cfg.yaml"
