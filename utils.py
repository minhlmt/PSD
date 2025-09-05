
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
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from PIL import Image
import urllib.request
import tarfile

from networks import ConvNet, ResNet, BasicBlock, AlexNet, VGG11, ViTModel
# from resnet import ResNet
from augment import DiffAug


class CenterCropLongEdgePIL:
    """
    Center crop the given PIL image according to the short edge,
    giữ phần giữa của ảnh.

    Args:
        keys (list[str], optional): danh sách keys nếu input là dict,
            hoặc None nếu trực tiếp PIL Image.
    """

    def __call__(self, img_or_dict):
        """
        Nếu input là dict, crop tất cả ảnh theo keys;
        Nếu input là PIL Image, crop trực tiếp.
        """
        if isinstance(img_or_dict, dict):
            for key in img_or_dict.keys():
                img_or_dict[key] = self._crop(img_or_dict[key])
            return img_or_dict
        else:
            return self._crop(img_or_dict)

    def _crop(self, img):
        """
        img: PIL.Image
        """
        w, h = img.size
        crop_size = min(w, h)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        return img.crop((left, top, right, bottom))

    def __repr__(self):
        return f"{self.__class__.__name__}()"



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
def download_imagenette(data_dir):
    """
    Download ImageNette v2 (160px) nếu chưa có và giải nén.
    """
    url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz'
    tgz_path = os.path.join(data_dir, 'imagenette2-160.tgz')
    extract_path = os.path.join(data_dir, 'imagenette2-160')

    if not os.path.exists(extract_path):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Downloading ImageNette from {url} ...")
        urllib.request.urlretrieve(url, tgz_path)
        print("Extracting...")
        with tarfile.open(tgz_path) as tar:
            tar.extractall(path=data_dir)
        print("Done.")
    return extract_path

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
    elif args.data == 'svhn':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                 split='train',
                                 download=True,
                                 transform=transform_train)
        testset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                split='test',
                                download=True,
                                transform=transform_test)
    
    elif args.data == 'imagenette':
        transform_train=transforms.Compose([ 
            CenterCropLongEdgePIL(), 
            transforms.Resize((128, 128)),
            transforms.ToTensor(), 
            transforms.Lambda(lambda x: x[[2,1,0],:,:]),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
        ])
        transform_test = transforms.Compose([
            CenterCropLongEdgePIL(),               # crop vuông theo cạnh ngắn nhất
            transforms.Resize((128,128)),          # resize về 128x128
            transforms.ToTensor(),                 # [0,1], [C,H,W]
            transforms.Lambda(lambda x: x[[2,1,0],:,:]),  # RGB -> BGR
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))  # [-1,1]
        ])
        data_path = download_imagenette(args.data_dir)
        trainset = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=transform_train)
        testset  = datasets.ImageFolder(os.path.join(data_path, 'val'),   transform=transform_test)

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
        model_name = e_model
    else:
        model_pool = ['convnet','resnet18']
        model_name = random.choice(model_pool)
        print('Random model: {}'.format(model_name))

    if args.data == 'imagenette':
        net_depth = 5
        im_size = (128, 128)
    else:
        im_size = (32, 32)

    if model_name == 'convnet':
        return ConvNet(channel = 3, num_classes= num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size), model_name
    elif model_name == 'resnet18':
        return ResNet(BasicBlock, [2,2,2,2], channel=3, num_classes=num_classes), model_name
    elif model_name == 'alexnet':
        return AlexNet(channel=3, num_classes=num_classes), model_name
    elif model_name == 'VGG11':
        return VGG11( channel=3, num_classes=num_classes), model_name
    elif model_name =='ViT':
        return ViTModel((32,32), 10), model_name

def remove_aug(augtype, remove_aug):
    aug_list = []
    for aug in augtype.split("_"):
        if aug not in remove_aug.split("_"):
            aug_list.append(aug)

    return "_".join(aug_list)


def diffaug(args, device=None):
    """Differentiable augmentation for condensation
    """
    if device is None:
        device = args.device
    aug_type = args.aug_type
    if args.data == 'cifar10':
        normalize = Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201), device=device)
    elif args.data == 'cifar100':
        normalize = Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201), device=device)
    elif args.data == 'svhn':
        normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), device=device)
    elif args.data == 'imagenette':
        normalize = Normalize((0.5,0.5,0.5), (0.5,0.5,0.5), device=device)
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
        for name, module in (modules):
            if name.startswith('layer4') and isinstance(module, nn.ReLU):
                hooks.append(module.register_forward_hook(getActivation(activations, name)))
    return hooks


def get_channel_attention(feature_set, exp=4, norm='l2'):
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

def prediction_matching_loss(real_logits, syn_logits, real_labels, temperature=4):
    """
    Tính prediction matching loss sử dụng KL-Divergence, với trung bình theo lớp và chuẩn hóa variance.

    Args:
        real_logits (torch.Tensor): Logits của tập dữ liệu thực (shape: [batch_size, num_classes]).
        syn_logits (torch.Tensor): Logits của tập dữ liệu tổng hợp (shape: [batch_size, num_classes]).
        real_labels (torch.Tensor): Nhãn của dữ liệu thực và tổng hợp (shape: [batch_size]).
        temperature (float): Tham số nhiệt độ cho softmax.

    Returns:
        torch.Tensor: KL-Divergence loss tính trên các phân phối trung bình theo lớp.
    """
    # Kiểm tra đầu vào
    if real_logits.shape != syn_logits.shape:
        raise ValueError(f"Kích thước real_logits {real_logits.shape} không khớp với syn_logits {syn_logits.shape}")
    if real_logits.shape[0] != real_labels.shape[0]:
        raise ValueError(f"Số mẫu của real_logits {real_logits.shape[0]} không khớp với labels {real_labels.shape[0]}")
    if torch.any(torch.isnan(real_logits)) or torch.any(torch.isinf(real_logits)):
        raise ValueError("real_logits chứa giá trị NaN hoặc Inf")
    if torch.any(torch.isnan(syn_logits)) or torch.any(torch.isinf(syn_logits)):
        raise ValueError("syn_logits chứa giá trị NaN hoặc Inf")
    if torch.any(real_labels < 0) or torch.any(real_labels >= real_logits.shape[1]):
        raise ValueError("real_labels chứa nhãn không hợp lệ")
    if real_logits.device != syn_logits.device or real_logits.device != real_labels.device:
        raise ValueError("real_logits, syn_logits và real_labels phải ở cùng device")

    # Chuẩn hóa logits
    real_logits = real_logits / (torch.sqrt(torch.var(real_logits, dim=0, keepdim=True) + 1e-4))
    syn_logits = syn_logits / (torch.sqrt(torch.var(syn_logits, dim=0, keepdim=True) + 1e-4))

    # Tính xác suất softmax
    real_probs = F.softmax(real_logits / temperature, dim=1)
    syn_probs = F.softmax(syn_logits / temperature, dim=1)

    # Tính trung bình xác suất theo lớp
    num_classes = real_logits.shape[1]
    one_hot_labels = F.one_hot(real_labels, num_classes=num_classes).float()
    class_counts = one_hot_labels.sum(dim=0)
    valid_classes = class_counts > 0
    if not valid_classes.any():
        raise ValueError("Không có lớp nào có mẫu hợp lệ trong real_labels")

    real_probs_mean = torch.matmul(one_hot_labels.t(), real_probs) / (class_counts.unsqueeze(1) + 1e-8)
    syn_probs_mean = torch.matmul(one_hot_labels.t(), syn_probs) / (class_counts.unsqueeze(1) + 1e-8)

    # Chỉ lấy các lớp hợp lệ
    real_probs_mean = real_probs_mean[valid_classes]
    syn_probs_mean = syn_probs_mean[valid_classes]

    # Kiểm tra phân phối xác suất
    if not torch.allclose(real_probs_mean.sum(dim=1), torch.ones_like(real_probs_mean.sum(dim=1)), atol=1e-4):
        raise ValueError("real_probs_mean không phải là phân phối xác suất hợp lệ")
    if not torch.allclose(syn_probs_mean.sum(dim=1), torch.ones_like(syn_probs_mean.sum(dim=1)), atol=1e-4):
        raise ValueError("syn_probs_mean không phải là phân phối xác suất hợp lệ")

    # Tính KL-Divergence
    syn_log_probs_mean = torch.log(syn_probs_mean + 1e-8)
    kl_loss = F.kl_div(syn_log_probs_mean, real_probs_mean, reduction='sum')

    return kl_loss

def test(args, model, testloader, criterion):
    '''Calculate accuracy
    '''
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (img, lab) in enumerate(testloader):
        img = img.to(args.device)
        lab = lab.to(args.device)

        with torch.no_grad():
            output = model(img)[1]
        loss = criterion(output, lab)
        acc1, acc5 = accuracy(output.data, lab, topk=(1, 5))
        losses.update(loss.item(), output.shape[0])
        top1.update(acc1.item(), output.shape[0])
        top5.update(acc5.item(), output.shape[0])

    return top1.avg, top5.avg, losses.avg

def validate(args, generator, testloader, criterion, aug_rand):
    '''Validate the generator performance
    '''
    all_best_top1 = []
    all_best_top5 = []
    for e_model in args.eval_model:
        print('Evaluating {}'.format(e_model))
        model, _ = define_model(args, args.num_classes, e_model)
        model = model.to(args.device)
        model.train()
        optim_model = torch.optim.SGD(model.parameters(), args.eval_lr, momentum=args.momentum,
                                      weight_decay=args.weight_decay)

        generator.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        best_top1 = 0.0
        best_top5 = 0.0



        for epoch_idx in range(args.epochs_eval):
            total_samples = args.ipc * args.num_classes  # Tổng số mẫu cần tạo
            lab_syn_full = torch.cat([torch.full((args.ipc,), i, dtype=torch.long)
                                      for i in range(args.num_classes)])  # Nhãn cố định cho mỗi lớp
           
            noise_full = torch.randn(total_samples, args.dim_noise)  # Nhiễu cho tất cả mẫu
            lab_syn_full = lab_syn_full.to(args.device)
            noise_full = noise_full.to(args.device)
            batch_size_gen = args.batch_size  
            with torch.no_grad():
                img_syn_full = generator(noise_full, lab_syn_full)  # Tạo tất cả hình ảnh tổng hợp một lần
                img_syn_full = aug_rand((img_syn_full + 1.0) / 2.0)
            for batch_idx in range((total_samples + args.batch_size - 1) // args.batch_size):
                start_idx = batch_idx * args.batch_size
                end_idx = min(start_idx + args.batch_size, total_samples)
                img_syn = img_syn_full[start_idx:end_idx]
                lab_syn = lab_syn_full[start_idx:end_idx]
               
                if np.random.rand(1) < args.mix_p and args.mixup_net == 'cut':
                    lam = np.random.beta(args.beta, args.beta)
                    rand_index = torch.randperm(len(img_syn)).to(args.device)

                    lab_syn_b = lab_syn[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(img_syn.size(), lam)
                    img_syn[:, :, bbx1:bbx2, bby1:bby2] = img_syn[rand_index, :, bbx1:bbx2, bby1:bby2]
                    ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img_syn.size()[-1] * img_syn.size()[-2]))

                    output = model(img_syn)[1]
                    loss = criterion(output, lab_syn) * ratio + criterion(output, lab_syn_b) * (1. - ratio)
                else:
                    output = model(img_syn)[1]
                    loss = criterion(output, lab_syn)

                acc1, acc5 = accuracy(output.data, lab_syn, topk=(1, 5))

                losses.update(loss.item(), img_syn.shape[0])
                top1.update(acc1.item(), img_syn.shape[0])
                top5.update(acc5.item(), img_syn.shape[0])

                optim_model.zero_grad()
                loss.backward()
                optim_model.step()

            if (epoch_idx + 1) % args.test_interval == 0:
                test_top1, test_top5, test_loss = test(args, model, testloader, criterion)
                print('[Test Epoch {}] Top1: {:.3f} Top5: {:.3f}'.format(epoch_idx + 1, test_top1, test_top5))
                if test_top1 > best_top1:
                    best_top1 = test_top1
                    best_top5 = test_top5

        all_best_top1.append(best_top1)
        all_best_top5.append(best_top5)

    return all_best_top1, all_best_top5

def train(args, epoch, generator, optim_g, trainloader, criterion, aug, aug_rand):
    '''The main training function for the generator with meta-learning integration'''
    activations = {}

    generator.train()
    gen_losses = AverageMeter()
    model, model_name = define_model(args, args.num_classes)
    model = model.to(args.device)
   
    optim_model = torch.optim.SGD(model.parameters(), args.eval_lr, momentum=args.momentum,
                                  weight_decay=args.weight_decay)
    if model_name =='convnet':
        model.load_state_dict(torch.load(args.weight_convnet, map_location=args.device))
    elif model_name =='resnet18':
        model.load_state_dict(torch.load(args.weight_resnet, map_location=args.device))
    # else:
    #      model.load_state_dict(torch.load('/kaggle/input/resnet34/resnet34_svhn.pth', map_location=args.device))
    model.train()
    for batch_idx, (img_real, lab_real) in enumerate(trainloader):
        img_real = img_real.to(args.device)
        lab_real = lab_real.to(args.device)
        gen_loss = torch.tensor(0.0, requires_grad=True).to(args.device)
        tl= torch.tensor(0.0, requires_grad=True).to(args.device)
        loss = torch.tensor(0.0, requires_grad=True).to(args.device)

        # 1. Train Generator
        optim_g.zero_grad()

        noise = torch.randn(args.batch_size, args.dim_noise).to(args.device)
        img_syn = generator(noise, lab_real)

        # 4. Attention-based loss
        hooks = attach_hooks(activations, model_name, model)
        output_real,  logit_real= model(img_real)
        activations, original_model_set_activations = refreshActivations(activations)
        delete_hooks(hooks)

        hooks = attach_hooks(activations, model_name, model)
        output_syn, logit_syn = model(img_syn)
        activations, syn_model_set_activations = refreshActivations(activations)
        delete_hooks(hooks)

        length_of_network = len(original_model_set_activations)
        for layer in range(length_of_network - 1):
            real_attention = get_combined_attention(original_model_set_activations[layer].detach())
            syn_attention = get_combined_attention(syn_model_set_activations[layer])
            real_attention = real_attention / (torch.sqrt(torch.var(real_attention, dim=0, keepdim=True) + 1e-8))
            syn_attention = syn_attention / (torch.sqrt(torch.var(syn_attention, dim=0, keepdim=True) + 1e-8))
            tl = error(real_attention, syn_attention,lab_real)

            loss+=tl

        # # 5. Output-based loss
        output_real = output_real / torch.sqrt(torch.var(output_real, dim=0, keepdim=True) + 1e-8)
        output_syn = output_syn / torch.sqrt(torch.var(output_syn, dim=0, keepdim=True) + 1e-8)
        
        output_loss =error(output_real, output_syn, lab_real)
       
        loss1= output_loss + loss
        gen_loss += loss1        
        pm_loss = prediction_matching_loss(logit_real.detach(), logit_syn, lab_real, temperature=4)
        gen_loss += 1000*pm_loss
        gen_loss.backward()
        optim_g.step()

        gen_losses.update(gen_loss.item())
        if (batch_idx + 1) % args.print_freq == 0:
            print('[Train Epoch {} Iter {}] G Loss: {:.3f}({:.3f})'.format(
                epoch, batch_idx + 1, gen_losses.val, gen_losses.avg))


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class TimeStamp():
    def __init__(self, print_log=True):
        self.prev = time.time()
        self.print_log = print_log
        self.times = {}

    def set(self):
        self.prev = time.time()

    def flush(self):
        if self.print_log:
            print("\n=========Summary=========")
            for key in self.times.keys():
                times = np.array(self.times[key])
                print(
                    f"{key}: {times.sum():.4f}s (avg {times.mean():.4f}s, std {times.std():.4f}, count {len(times)})"
                )
                self.times[key] = []

    def stamp(self, name=''):
        if self.print_log:
            spent = time.time() - self.prev
            # print(f"{name}: {spent:.4f}s")
            if name in self.times.keys():
                self.times[name].append(spent)
            else:
                self.times[name] = [spent]
            self.set()


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Plotter():
    def __init__(self, path, nepoch, idx=0):
        self.path = path
        self.data = {'epoch': [], 'acc_tr': [], 'acc_val': [], 'loss_tr': [], 'loss_val': []}
        self.nepoch = nepoch
        self.plot_freq = 10
        self.idx = idx

    def update(self, epoch, acc_tr, acc_val, loss_tr, loss_val):
        self.data['epoch'].append(epoch)
        self.data['acc_tr'].append(acc_tr)
        self.data['acc_val'].append(acc_val)
        self.data['loss_tr'].append(loss_tr)
        self.data['loss_val'].append(loss_val)

        if len(self.data['epoch']) % self.plot_freq == 0:
            self.plot()

    def plot(self, color='black'):
        fig, axes = plt.subplots(1, 4, figsize=(4 * 4, 3))
        fig.tight_layout(h_pad=3, w_pad=3)

        fig.suptitle(f"{self.path}", size=16, y=1.1)

        axes[0].plot(self.data['epoch'], self.data['acc_tr'], color, lw=0.8)
        axes[0].set_xlim([0, self.nepoch])
        axes[0].set_ylim([0, 100])
        axes[0].set_title('acc train')

        axes[1].plot(self.data['epoch'], self.data['acc_val'], color, lw=0.8)
        axes[1].set_xlim([0, self.nepoch])
        axes[1].set_ylim([0, 100])
        axes[1].set_title('acc val')

        axes[2].plot(self.data['epoch'], self.data['loss_tr'], color, lw=0.8)
        axes[2].set_xlim([0, self.nepoch])
        axes[2].set_ylim([0, 3])
        axes[2].set_title('loss train')

        axes[3].plot(self.data['epoch'], self.data['loss_val'], color, lw=0.8)
        axes[3].set_xlim([0, self.nepoch])
        axes[3].set_ylim([0, 3])
        axes[3].set_title('loss val')

        for ax in axes:
            ax.set_xlabel('epochs')

        plt.savefig(f'{self.path}/curve_{self.idx}.png', bbox_inches='tight')
        plt.close()


def random_indices(y, nclass=10, intraclass=False, device='cpu'):
    n = len(y)
    if intraclass:
        index = torch.arange(n).to(device)
        for c in range(nclass):
            index_c = index[y == c]
            if len(index_c) > 0:
                randidx = torch.randperm(len(index_c))
                index[y == c] = index_c[randidx]
    else:
        index = torch.randperm(n).to(device)
    return index


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""
    def __init__(self, alphastd, eigval, eigvec, device='cpu'):
        self.alphastd = alphastd
        self.eigval = torch.tensor(eigval, device=device)
        self.eigvec = torch.tensor(eigvec, device=device)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        # make differentiable
        if len(img.shape) == 4:
            return img + rgb.view(1, 3, 1, 1).expand_as(img)
        else:
            return img + rgb.view(3, 1, 1).expand_as(img)


class Grayscale(object):
    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img):
        self.transforms = []
        if self.brightness != 0:
            self.transforms.append(Brightness(self.brightness))
        if self.contrast != 0:
            self.transforms.append(Contrast(self.contrast))
        if self.saturation != 0:
            self.transforms.append(Saturation(self.saturation))

        random.shuffle(self.transforms)
        transform = Compose(self.transforms)
        # print(transform)
        return transform(img)


class CutOut():
    def __init__(self, ratio, device='cpu'):
        self.ratio = ratio
        self.device = device

    def __call__(self, x):
        n, _, h, w = x.shape
        cutout_size = [int(h * self.ratio + 0.5), int(w * self.ratio + 0.5)]
        offset_x = torch.randint(h + (1 - cutout_size[0] % 2), size=[1], device=self.device)[0]
        offset_y = torch.randint(w + (1 - cutout_size[1] % 2), size=[1], device=self.device)[0]

        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(n, dtype=torch.long, device=self.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=self.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=self.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=h - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=w - 1)
        mask = torch.ones(n, h, w, dtype=x.dtype, device=self.device)
        mask[grid_batch, grid_x, grid_y] = 0

        x = x * mask.unsqueeze(1)
        return x


class Normalize():
    def __init__(self, mean, std, device='cpu'):
        self.mean = torch.tensor(mean, device=device).reshape(1, len(mean), 1, 1)
        self.std = torch.tensor(std, device=device).reshape(1, len(mean), 1, 1)

    def __call__(self, x, seed=-1):
        return (x - self.mean) / self.std
