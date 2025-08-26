import argparse
import torch
import random
import numpy as np
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
    parser.add_argument('--eval_model', type=str, default=['convnet'], help='the model to evaluate, e.g., convnet/resnet18/resnet50/alexnet/vgg11/vit')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for data loading')
    parser.add_argument('--print_freq', type=int, default=50, help='print frequency during training')
    parser.add_argument('--eval_interval', type=int, default=5, help='interval for evaluating the model during training')
    parser.add_argument('--test_interval', type=int, default=200, help='interval for testing the model during training')
    parser.add_argument('--data', type=str, default='cifar10', help='dataset')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--data_dir', type=str, default='./data', help='dataset directory')
    parser.add_argument('--output_dir', type=str, default='./results', help='output directory')
    parser.add_argument('--logs_dir', type=str, default='./logs', help='logs directory')
    parser.add_argument('--aug_type', type=str, default='color_crop_cutout', help='augmentation type')
    parser.add_argument('--mixup_net', type=str, default='cut', help='')
    parser.add_argument('--mix-p', type=float, default=-1.0)
    parser.add_argument('--seed', type=int, default=3407)    
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

if __name__ == '__main__':
    main()