import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging, numpy

logger = logging.getLogger('Edge.Filter')

FILTER_LIST = ['average', 'gaussian', 'median', 'none', 'averageLOW']


# 1. Average Filter (Mean Filter)
def average_low_filter(x, kernel_size = 3):
    with torch.no_grad():
        channels = x.shape[1]   # (B, C, H, W)
        kernel = torch.ones((channels, 1, kernel_size, kernel_size)) / (kernel_size ** 2)
        kernel = kernel.to(x.device)
        padding = kernel_size // 2
        low_pass_filtered = F.conv2d(x, kernel, padding=padding, groups=channels)
    return low_pass_filtered

def average_high_filter(x, kernel_size = 3):
    low_x = average_low_filter(x, kernel_size=kernel_size)
    x = x - low_x.detach()
    return x

def average1d_low_filter(x, kernel_size = 3):
    with torch.no_grad():
        B,C,L = x.shape   # (B, C, L)
        kernel = torch.ones((1, 1, kernel_size)) / kernel_size
        kernel = kernel.to(x.device)
        x = x.reshape(B*C, 1, L)
        padding = kernel_size // 2
        low_pass_filtered = F.conv1d(x, kernel, padding=padding).view(B, C, L)
    return low_pass_filtered

def average1d_high_filter(x, kernel_size = 3):
    low_x = average1d_low_filter(x, kernel_size=kernel_size)
    x = x - low_x.detach()
    return x


# 2. Gaussian Filter
def gaussian_low_filter(x, kernel_size = 3, sigma = 1.0):
    with torch.no_grad():
        channels = x.shape[1]
        a = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
        b = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
        bb, aa = torch.meshgrid(b, a, indexing="ij")
        kernel = torch.exp(-(aa**2 + bb**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        kernel = kernel.view(1, 1, kernel_size, kernel_size)  # [1, 1, kernel_height, kernel_width]
        kernel = kernel.repeat(channels, 1, 1, 1)  # [channels, 1, kernel_height, kernel_width]
        kernel = kernel.to(x.device)

        padding = kernel_size // 2
        low_pass_filtered = F.conv2d(x, kernel, padding=padding, groups=channels)
    return low_pass_filtered

def gaussian_high_filter(x, kernel_size = 3, sigma = 1.0):
    low_x = gaussian_low_filter(x, kernel_size=kernel_size, sigma=sigma)
    x = x - low_x.detach()
    return x

def gaussian1d_low_filter(x, kernel_size = 3, sigma = 1.0):
    with torch.no_grad():
        B, C, L = x.shape
        device = x.device

        mean = (kernel_size - 1) / 2.0
        tmp = torch.arange(kernel_size, dtype=torch.float32, device=device)
        gaussian = torch.exp(-0.5 * ((tmp - mean) / sigma)**2)
        kernel = gaussian / gaussian.sum()
        kernel = kernel.view(1, 1, kernel_size)
        x = x.reshape(B * C, 1, L)
        padding = kernel_size // 2
        low_pass_filtered = F.conv1d(x, kernel, padding=padding).view(B, C, L)
    return low_pass_filtered

def gaussian1d_high_filter(x, kernel_size = 3, sigma = 1.0):
    low_x = gaussian1d_low_filter(x, kernel_size=kernel_size, sigma=sigma)
    x = x - low_x.detach()
    return x


# 3.Median Filter
def median_low_filter(x, kernel_size = 3):
    with torch.no_grad():
        channels = x.shape[1]   # (B, C, H, W)
        pad = kernel_size // 2
        img_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        # unfold: (B, C, H, W) -> (B, C, H, W, k, k)
        patches = img_padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
        # patches: (B, C, H, W, k, k) -> (B, C, H, W, k*k)
        patches = patches.contiguous().view(*patches.size()[:4], -1)
        # median: (B, C, H, W)
        median = patches.median(dim=-1)[0]
    return median

def median_high_filter(x, kernel_size = 3):
    low_x = median_low_filter(x, kernel_size=kernel_size)
    x = x - low_x.detach()
    return x

def median1d_low_filter(x, kernel_size = 3):
    with torch.no_grad():
        pad = kernel_size // 2
        x_padded = F.pad(x, (pad, pad), mode='reflect')
        # 1D unfold: (B, C, L) -> (B, C, L, K)
        patches = x_padded.unfold(dimension=2, size=kernel_size, step=1)
        # median 계산: (B, C, L)
        low_pass_filtered = patches.median(dim=-1)[0]
    return low_pass_filtered

def median1d_high_filter(x, kernel_size = 3):
    low_x = median1d_low_filter(x, kernel_size=kernel_size)
    x = x - low_x.detach()
    return x


class FilterBase():
    def __init__(self, cfg, mode, position):
        assert mode in FILTER_LIST
        self.input_density = []
        self.output_density = []

        kernel_size = cfg.MODEL.FILTER.KERNEL
        
        if mode == 'average':
            self.filterLayer = lambda x: average_high_filter(x, kernel_size)
            logger.info(f'Average Filter with kernel size: {kernel_size} applied at position {position}')
        elif mode == 'averageLOW':
            self.filterLayer = lambda x: average_low_filter(x, kernel_size)
            logger.info(f'AverageLOW Filter with kernel size: {kernel_size} applied at position {position}')
        elif mode == 'median':
            self.filterLayer = lambda x: median_high_filter(x, kernel_size)
            logger.info(f'Median Filter with kernel size: {kernel_size} applied at position {position}')
        elif mode == 'gaussian':
            self.filterLayer = lambda x: gaussian_high_filter(x, kernel_size)
            logger.info(f'Gaussian Filter with kernel size: {kernel_size} applied at position {position}')
        elif mode == 'none':
            self.filterLayer = lambda x: x
            logger.info(f'No Filter applied')

    def reset(self):
        self.input_density = []
        self.output_density = []

    def __str__(self):
        return f'Input density: {numpy.array(self.input_density).mean():.4f}, Output density: {numpy.array(self.output_density).mean():.4f}'
    
    def log(self):
        return {'Input_density': numpy.array(self.input_density).mean(), 'Output_density': numpy.array(self.output_density).mean()}

    def __call__(self, x, verbose=False):
        out = self.filterLayer(x)
        self.input_density.append((x > 0).sum().cpu().detach().numpy() / x.numel())
        self.output_density.append((out > 0).sum().cpu().detach().numpy() / out.numel())
        return out

class EdgeWRN(nn.Module):
    def __init__(self, cfg, base_model):
        super().__init__()
        assert cfg.MODEL.FILTER.POSITION in [0, 1, 2, 3]

        self.conv1 = base_model.conv1
        self.block1 = base_model.block1
        if hasattr(base_model, 'sub_block1'):
            self.sub_block1 = base_model.sub_block1
        self.block2 = base_model.block2
        self.block3 = base_model.block3
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.fc = base_model.fc
        self.nChannels = base_model.nChannels

        self.position = cfg.MODEL.FILTER.POSITION
        self.filter = FilterBase(cfg, cfg.MODEL.FILTER.MODE, cfg.MODEL.FILTER.POSITION)

        self.before_filter = None
        self.after_filter = None

    def forward(self, x, return_features=False):
        out = x
        layers = [
            self.conv1,
            self.block1,
            self.block2,
            self.block3
        ]
        for i, layer in enumerate(layers):
            out = layer(out)
            if i == self.position:
                out = self.filter(out)

        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        if return_features:
            features = out.clone().detach()
            return self.fc(out), features
        return self.fc(out)

class EdgeResNet(nn.Module):
    def __init__(self, cfg, base_model):
        super().__init__()
        assert cfg.MODEL.FILTER.POSITION in [0, 1, 2, 3, 4]

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        self.linear = base_model.linear

        self.position = cfg.MODEL.FILTER.POSITION
        self.filter = FilterBase(cfg, cfg.MODEL.FILTER.MODE, cfg.MODEL.FILTER.POSITION)

    def forward(self, x):
        out = x
        layers = [
            self.conv1,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4
        ]

        for i, layer in enumerate(layers):
            out = layer(out)
            if i == self.position:
                out = self.filter(out)
            if i==0:
                out = F.relu(self.bn1(out))

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


#########################
####    ViT Filter   ####
#########################

class ViTFilterBase():
    def __init__(self, cfg, mode, position):
        assert mode in FILTER_LIST
        self.input_density = []
        self.output_density = []

        kernel_size = cfg.MODEL.FILTER.KERNEL

        if mode == 'average':
            self.filterLayer = lambda x, n_h, n_w: filter_to_1d(x, average1d_high_filter, kernel_size)
            logger.info(f'Average 1-dim Filter with kernel size: {kernel_size} applied at position {position}')
        elif mode == 'averageLOW':
            self.filterLayer = lambda x, n_h, n_w: filter_to_1d(x, average1d_low_filter, kernel_size)
            logger.info(f'AverageLOW 1-dim Filter with kernel size: {kernel_size} applied at position {position}')
        elif mode == 'gaussian':
            self.filterLayer = lambda x, n_h, n_w: filter_to_1d(x, gaussian1d_high_filter, kernel_size)
            logger.info(f'Gaussian 1-dim Filter with kernel size: {kernel_size} applied at position {position}')
        elif mode == 'median':
            self.filterLayer = lambda x, n_h, n_w: filter_to_1d(x, median1d_high_filter, kernel_size)
            logger.info(f'Median 1-dim Filter with kernel size: {kernel_size} applied at position {position}')
        elif mode == 'none':
            self.filterLayer = lambda x, n_h, n_w: x
            logger.info(f'No Filter applied')

    def reset(self):
        self.input_density = []
        self.output_density = []

    def __str__(self):
        return f'Input density: {numpy.array(self.input_density).mean():.4f}, Output density: {numpy.array(self.output_density).mean():.4f}'
    
    def log(self):
        return {'Input_density': numpy.array(self.input_density).mean(), 'Output_density': numpy.array(self.output_density).mean()}

    def __call__(self, x, n_h, n_w):
        out = self.filterLayer(x, n_h, n_w)
        self.input_density.append((x > 0).sum().cpu().detach().numpy() / x.numel())
        self.output_density.append((out > 0).sum().cpu().detach().numpy() / out.numel())
        return out

def filter_to_1d(x, filter, kernel_size):
    x = x.permute(0,2,1)    # (B,C,L)
    x = filter(x, kernel_size)
    x = x.permute(0,2,1)
    return x


class EdgeViT(nn.Module):
    def __init__(self, cfg, base_model):
        super().__init__()
        self.cfg = cfg
        self.dropout = base_model.dropout
        self.conv_proj = base_model.conv_proj
        self.encoder = base_model.encoder
        self.heads = base_model.heads

        self.patch_size = base_model.patch_size
        self.hidden_dim = base_model.hidden_dim
        self.num_classes = base_model.num_classes
        self.class_token = nn.Parameter(torch.zeros(1, 1, base_model.hidden_dim))

        self.position = cfg.MODEL.FILTER.POSITION
        self.filter = ViTFilterBase(cfg, cfg.MODEL.FILTER.MODE, cfg.MODEL.FILTER.POSITION)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)

        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x, n_h, n_w

    def forward(self, x, return_features=False):
        # Reshape and permute the input tensor
        x, n_h, n_w = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = x + self.encoder.pos_embedding
        x = self.encoder.dropout(x)
        for idx, layer in enumerate(self.encoder.layers):   # 0~11
            x = layer(x)
            if idx == self.position:
                x = self.filter(x, n_h, n_w)
        x = self.encoder.ln(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        if return_features:
            features = x.clone().detach()
            x = self.heads(x)
            return x, features

        x = self.heads(x)
        return x
