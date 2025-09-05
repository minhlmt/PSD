import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from mmgen.models import build_model
from model_biggan import model_biggan, train_cfg, test_cfg, optimizer

# from resnet import ResNet
from utils import Logger, load_data, diffaug, train, validate
from augment import DiffAug


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for training & evaluating networks')
    parser.add_argument('--epochs', type=int, default=300, help='epochs to train a model with real data')
    parser.add_argument('--epochs_eval', type=int, default=3000, help='epochs to train a model with synthetic data') 
    parser.add_argument('--epochs_match_train', type=int, default=3000, help='epochs to train a model with synthetic data') 
    parser.add_argument('--eval_lr', type=int, default=0.01, help='learning rate for evaluating networks')
    parser.add_argument('--momentum', type=int, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', type=int, default=5e-4, help='weight decay for SGD')
    parser.add_argument('--eval_model', nargs='+', default=['convnet'], help='the model to evaluate, e.g., convnet/resnet18/resnet50/alexnet/vgg11/vit')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for data loading')
    parser.add_argument('--print_freq', type=int, default=50, help='print frequency during training')
    parser.add_argument('--eval_interval', type=int, default=5, help='interval for evaluating the model during training')
    parser.add_argument('--test_interval', type=int, default=200, help='interval for testing the model during training')
    parser.add_argument('--data', type=str, default='cifar10', help='dataset')
    parser.add_argument('--weight_biggan', type=str, default='./pretrain_model/cifar/biggan/biggan_cifar10.pth', help='the path to load the pretrained BigGAN')
    parser.add_argument('--weight_convnet', type=str, default='./pretrain_model/cifar/convnet/convnet_cifar10.pth', help='the path to load the pretrained convnet')
    parser.add_argument('--weight_resnet', type=str, default='./pretrain_model/cifar/resnet18/resnet18_cifar10.pth', help='the path to load the pretrained resnet18')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--data_dir', type=str, default='./data', help='dataset directory')
    parser.add_argument('--output_dir', type=str, default='./results', help='output directory')
    parser.add_argument('--logs_dir', type=str, default='./logs', help='logs directory')
    parser.add_argument('--aug_type', type=str, default='color_crop_cutout', help='augmentation type')
    parser.add_argument('--mixup_net', type=str, default='cut', help='')
    parser.add_argument('--mix-p', type=float, default=-1.0)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--tag', type=str, default='', help='tag for output directory')
    parser.add_argument('--dim_noise', type=int, default=128, help='dimension of noise vector')
    parser.add_argument('--beta', type=float, default=1.0, help='beta parameter for mixup')
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_dir = args.output_dir + args.tag
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.output_dir + '/outputs'):
        os.makedirs(args.output_dir + '/outputs')

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    args.logs_dir = args.logs_dir + args.tag
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    sys.stdout = Logger(os.path.join(args.logs_dir, 'logs.txt'))

    print(args)


    model_g = build_model(model_biggan, train_cfg=train_cfg, test_cfg=test_cfg)
    checkpoint = torch.load(args.weight_biggan)
    state_dict = checkpoint.get('state_dict', checkpoint)
    model_g.load_state_dict(state_dict)
    generator = model_g.generator_ema.to(args.device)
    
    optim_g = torch.optim.Adam(generator.parameters(), lr=0.00002, betas=(0.0, 0.999))
    for g in optim_g.param_groups:
        g['lr'] = 0.00002




    trainloader, testloader = load_data(args)

    criterion = nn.CrossEntropyLoss()

    aug, aug_rand = diffaug(args)

    best_top1s = np.zeros((len(args.eval_model),))
    best_top5s = np.zeros((len(args.eval_model),))
    best_epochs = np.zeros((len(args.eval_model),))
    for epoch in range(args.epochs):
        generator.train()
        train(args, epoch, generator, optim_g, trainloader, criterion, aug, aug_rand)

        # save image for visualization
        generator.eval()
        test_label = torch.tensor(list(range(10)) * 10).to(args.device)
        test_noise  = torch.randn(50, args.dim_noise).to(args.device)
        test_img_syn = (generator(test_noise,test_label) + 1.0) / 2.0
        test_img_syn = make_grid(test_img_syn, nrow=10)
        save_image(test_img_syn, os.path.join(args.output_dir, 'outputs/img_{}.png'.format(epoch)))
        generator.train()

        if (epoch + 1) % args.eval_interval == 0:
            top1s, top5s = validate(args, generator, testloader, criterion, aug_rand)
            for e_idx, e_model in enumerate(args.eval_model):
                if top1s[e_idx] > best_top1s[e_idx]:
                    best_top1s[e_idx] = top1s[e_idx]
                    best_top5s[e_idx] = top5s[e_idx]
                    best_epochs[e_idx] = epoch

                    model_dict = {'generator': generator.state_dict(),
                                'optim_g': optim_g.state_dict()}
                    torch.save(
                        model_dict,
                        os.path.join(args.output_dir, 'model_dict_{}.pth'.format(e_model)))
                    print('Save model for {}'.format(e_model))

                print('Current Best Epoch for {}: {}, Top1: {:.3f}, Top5: {:.3f}'.format(e_model, best_epochs[e_idx], best_top1s[e_idx], best_top5s[e_idx]))

if __name__ == '__main__':
    main()