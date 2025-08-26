
import os
import sys
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from networks import ConvNet, ResNet,BasicBlock,AlexNet,VGG11,ViTModel
# from resnet import ResNet
from utils import AverageMeter, accuracy, Normalize, Logger, rand_bbox
from augment import DiffAug

def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling
def str2bool(v):
    """Cast string to boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_data(args):
    '''Obtain data
    '''
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if args.data == 'cifar10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201))
        ])
        trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root=args.data_dir, train=False, download=True,
                                   transform=transform_test)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )

    return trainloader, testloader



def define_model(args, num_classes, e_model=None):
    '''Obtain model for training, validating and matching
    With no 'e_model' specified, it returns a random model
    '''
    net_width, net_depth, net_act, net_norm, net_pooling= get_default_convnet_setting()
    if e_model:
        model = e_model
    else:
        model_pool = ['convnet','resnet18']
        model = random.choice(model_pool)
        print('Random model: {}'.format(model))
        args.net = model

    if args.data == 'mnist' or args.data == 'fashion':
        nch = 1
    else:
        nch = 3

    if model == 'convnet':
        return ConvNet(channel = 3, num_classes= num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=(32,32))
    elif model == 'resnet18':
        return ResNet(BasicBlock, [2,2,2,2], channel=3, num_classes=num_classes)
    elif model == 'alexnet':
        return AlexNet(channel=3, num_classes=num_classes)
    elif model == 'VGG11':
        return VGG11( channel=3, num_classes=num_classes)
    elif model =='ViT':
        return ViTModel((32,32), 10)

def remove_aug(augtype, remove_aug):
    aug_list = []
    for aug in augtype.split("_"):
        if aug not in remove_aug.split("_"):
            aug_list.append(aug)

    return "_".join(aug_list)


def diffaug(args, device):
    """Differentiable augmentation for condensation
    """
    aug_type = args.aug_type
    if args.data == 'cifar10':
        normalize = Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201), device = device)
    elif args.data == 'svhn':
        normalize = Normalize((0.437, 0.444, 0.473), (0.198, 0.201, 0.197), device = device)
    elif args.data == 'fashion':
        normalize = Normalize((0.286,), (0.353,), device = device)
    elif args.data == 'mnist':
        normalize = Normalize((0.131,), (0.308,), device = device)
    print("Augmentataion Matching: ", aug_type)
    augment = DiffAug(strategy=aug_type, batch=True)
    aug_batch = transforms.Compose([normalize, augment])

    if args.mixup_net == 'cut':
        aug_type = remove_aug(aug_type, 'cutout')
    print("Augmentataion Net update: ", aug_type)
    augment_rand = DiffAug(strategy=aug_type, batch=False)
    aug_rand = transforms.Compose([normalize, augment_rand])

    return aug_batch, aug_rand


def getActivation(activations,name):
    def hook_func(m, inp, op):
        activations[name] = op.clone()
    return hook_func

''' Defining the Refresh Function to store Activations and reset Collection '''
def refreshActivations(activations):
    model_set_activations = [] # Jagged Tensor Creation
    for i in activations.keys():
        model_set_activations.append(activations[i])
    activations = {}
    return activations, model_set_activations

''' Defining the Delete Hook Function to collect Remove Hooks '''
def delete_hooks(hooks):
    for i in hooks:
        i.remove()
    return


def attach_hooks(activations, model_name , model):
    hooks = []
    base = model.module if torch.cuda.device_count() > 1 else model
    if model_name == 'convnet':
        modules = base.features.named_modules()  # Nếu là ConvNet, chỉ lấy features
       
        for module in (modules):
            if isinstance(module[1], nn.ReLU):
                # Hook the Ouptus of a ReLU Layer
                hooks.append(base.features[int(module[0])].register_forward_hook(getActivation(activations,'ReLU_'+str(len(hooks)))))

    else:
        modules = base.named_modules()  # Nếu là ResNet hoặc model khác, lấy toàn bộ layers
        def hook_initial_conv(activations):
            def hook_func(module, input, output):
                activations['initial_conv'] = F.relu(output)
            return hook_func
    
        # Hook cho avg_pool (sau layer4)
        def hook_avg_pool(activations):
            def hook_func(module, input, output):
                pooled = F.avg_pool2d(output, 4)  # Giữ 4 như trong ResNet của bạn
                activations['avg_pool'] = pooled
            return hook_func
        for name,base in (modules):
            if name.startswith('layer4') and isinstance(base, nn.ReLU):
                hooks.append(module.register_forward_hook(getActivation(name)))
    return hooks


def get_channel_attention(feature_set, exp=6, norm='l2'):
    # Tổng giá trị trên không gian theo từng kênh (dim=[2, 3])
    # channel_attention = torch.sum(feature_set, dim=(2, 3))  # Kích thước: [B, C]

    # # Bình phương giá trị
    # channel_attention = channel_attention**exp  # Lũy thừa với `exp`
    powered_features = torch.abs(feature_set) ** exp  # [B, C, H, W]

    # Tổng hợp trên chiều không gian (H, W)
    channel_attention = torch.sum(powered_features, dim=(2, 3))  # [B, C]

    # Chuẩn hóa
    if norm == 'l2':
        # Chuẩn hóa bằng norm L2
        normalized_attention_maps = F.normalize(channel_attention, p=2.0, dim=1)
    elif norm == 'l1':
        # Chuẩn hóa bằng norm L1
        normalized_attention_maps = F.normalize(channel_attention, p=1.0, dim=1)
    elif norm == 'fro':
        # Chuẩn hóa Frobenius (norm tổng thể)
        fro_norm = torch.sqrt(torch.sum(channel_attention**2, dim=1))  # Kích thước: [B]
        normalized_attention_maps = channel_attention / fro_norm.unsqueeze(-1)
    elif norm == 'none':
        # Không chuẩn hóa
        normalized_attention_maps = channel_attention

    return normalized_attention_maps
def get_attention(feature_set, param=1, exp=1, norm='l2'):
    # Tính attention map ban đầu theo param
    if param == 0:
        attention_map = torch.sum(torch.abs(feature_set), dim=1)  # [B, H, W]
    elif param == 1:
        attention_map = torch.sum(torch.abs(feature_set) ** exp, dim=1)  # [B, H, W]
    elif param == 2:
        attention_map = torch.max(torch.abs(feature_set) ** exp, dim=1).values  # [B, H, W]

    # Không reshape về vector nữa — giữ nguyên [B, H, W]
    if norm == 'l2':
        # Normalize theo toàn bộ H×W cho từng ảnh
        flat = attention_map.view(attention_map.size(0), -1)
        normed = F.normalize(flat, p=2.0, dim=1)
        attention_map = normed.view_as(attention_map)

    return attention_map  # [B, H, W]

def get_combined_attention(feature_set, param=1, exp_spatial=2, exp_channel=6, norm='l2'):
    # Attention theo không gian (H x W), kết quả là [B, H, W]
    spatial_attention_map = get_attention(feature_set, param=param, exp=exp_spatial, norm=norm)  # [B, H, W]

    # Tổng theo từng cột (chiều H) → vector [B, W]
    spatial_attention_vector = torch.sum(spatial_attention_map, dim=1)  # [B, W]

    # Attention theo channel, kết quả là [B, C]
    channel_attention = get_channel_attention(feature_set, exp=exp_channel, norm=norm)  # [B, C]

    # Ghép channel attention với spatial column-wise vector attention
    combined_attention = torch.cat([channel_attention, spatial_attention_vector], dim=1)  # [B, C + W]

    return combined_attention


def error(real_feature, syn_feature, real_labels):
    """
    Tính MSE loss giữa đặc trưng trung bình theo lớp của real và syn, sử dụng vector hóa.

    Args:
        real_feature: Tensor có dạng (batch_size, D) hoặc (batch_size, C, H, W), đặc trưng của dữ liệu thực.
        syn_feature: Tensor có dạng (batch_size, D) hoặc (batch_size, C, H, W), đặc trưng của dữ liệu tổng hợp.
        real_labels: Tensor có dạng (batch_size,), nhãn lớp cho cả real_feature và syn_feature.

    Returns:
        MSE loss giữa đặc trưng trung bình theo lớp của real và syn, tính trên chiều D hoặc C.
    """
    MSE_Loss = nn.MSELoss(reduction='sum')

    # Kiểm tra kích thước
    if real_feature.shape != syn_feature.shape:
        raise ValueError(f"Kích thước real_feature {real_feature.shape} không khớp với syn_feature {syn_feature.shape}")
    if real_feature.shape[0] != real_labels.shape[0]:
        raise ValueError(f"Số mẫu của real_feature {real_feature.shape[0]} không khớp với real_labels {real_labels.shape[0]}")
    if real_feature.device != syn_feature.device or real_feature.device != real_labels.device:
        raise ValueError("real_feature, syn_feature và real_labels phải ở cùng device")

    # Xác định dạng đầu vào và kích thước đặc trưng
    if len(real_feature.shape) == 2:  # Dạng (batch_size, D)
        feature_dim = real_feature.shape[1]  # D
        is_2d = True
    elif len(real_feature.shape) == 4:  # Dạng (batch_size, C, H, W)
        feature_dim = real_feature.shape[1]  # C
        is_2d = False
    else:
        raise ValueError("Đầu vào phải có dạng (batch_size, D) hoặc (batch_size, C, H, W)")

    # Lấy số lớp tối đa từ kích thước của real_labels
    num_classes = int(real_labels.max().item()) + 1

    # Tạo ma trận one-hot từ real_labels
    one_hot_labels = F.one_hot(real_labels, num_classes=num_classes).float()  # [batch_size, num_classes]

    # Tính số lượng mẫu mỗi lớp
    class_counts = one_hot_labels.sum(dim=0)  # [num_classes]

    # Xử lý trường hợp 4D: tính trung bình trên H, W trước
    if not is_2d:
        real_feature = torch.mean(real_feature, dim=(2, 3))  # [batch_size, C]
        syn_feature = torch.mean(syn_feature, dim=(2, 3))    # [batch_size, C]

    # Tính tổng đặc trưng theo lớp bằng phép nhân ma trận
    real_feature_sum = torch.matmul(one_hot_labels.t(), real_feature)  # [num_classes, D] hoặc [num_classes, C]
    syn_feature_sum = torch.matmul(one_hot_labels.t(), syn_feature)    # [num_classes, D] hoặc [num_classes, C]

    # Tính trung bình đặc trưng theo lớp
    real_feature_mean = real_feature_sum / (class_counts.unsqueeze(1) + 1e-8)  # [num_classes, D] hoặc [num_classes, C]
    syn_feature_mean = syn_feature_sum / (class_counts.unsqueeze(1) + 1e-8)    # [num_classes, D] hoặc [num_classes, C]

    # Lấy các lớp hợp lệ (có mẫu)
    valid_classes = class_counts > 0
    if not valid_classes.any():
        return torch.tensor(0.0, device=real_feature.device, requires_grad=True)

    # Chỉ lấy đặc trưng trung bình của các lớp hợp lệ
    real_feature_mean = real_feature_mean[valid_classes]  # [num_valid_classes, D] hoặc [num_valid_classes, C]
    syn_feature_mean = syn_feature_mean[valid_classes]    # [num_valid_classes, D] hoặc [num_valid_classes, C]

    # Tính MSE loss
    mse = MSE_Loss(real_feature_mean, syn_feature_mean)
    return mse
