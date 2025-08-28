import argparse
import torch
import random
import numpy as np


import os
import sys
import time
import random
import argparse
import numpy as np
import urllib.request
import tarfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from networks import ConvNet, ResNet,BasicBlock,AlexNet,VGG11
# from resnet import ResNet
from utils import get_default_convnet_setting, AverageMeter, accuracy, Normalize, Logger, rand_bbox
from augment import DiffAug






def getActivation(activations,name):
    def hook_func(m, inp, op):
        activations[name] = op.clone().detach()
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

IMAGENETTE_TO_IMAGENET = {
    0: 0,     # n01440764 -> "tench" (class 0 trong ImageNet)
    1: 217,   # n02102040 -> "English springer"
    2: 482,   # n02979186 -> "cassette player"
    3: 491,   # n03000684 -> "chain saw"
    4: 497,   # n03028079 -> "church"
    5: 566,   # n03394916 -> "French horn"
    6: 569,   # n03417042 -> "garbage truck"
    7: 571,   # n03425413 -> "gas pump"
    8: 574,   # n03445777 -> "golf ball"
    9: 701    # n03888257 -> "parachute"
}
def train(args, epoch, generator, discriminator, optim_g, optim_d, trainloader, criterion, aug, aug_rand):
    '''The main training function for the generator with meta-learning integration'''
    activations = {}

    generator.train()
    gen_losses = AverageMeter()
    model = define_model(args, args.num_classes).to(args.device)
   
    optim_model = torch.optim.SGD(model.parameters(), args.eval_lr, momentum=args.momentum,
                                  weight_decay=args.weight_decay)
    # if args.net =='convnet':
    #     # model=torch.load('/kaggle/input/pretrain-model/convnet.pth')
    #     model.load_state_dict(torch.load('/kaggle/input/pretrain-model-imagenette-final/convnet_imagenette_final.pth', map_location=args.device))
    #     # teacher_model = torch.load('/kaggle/input/pretrain-model/convnet.pth').to(args.device)
    # elif args.net =='resnet18':
    #     # model=torch.load('/kaggle/input/pretrain-model/resnet_18.pth')
    #     model.load_state_dict(torch.load('/kaggle/input/pretrain-model-imagenette-final/resnet18_imagenette_final.pth', map_location=args.device))
    #     # teacher_model = torch.load('/kaggle/input/pretrain-model/resnet_18.pth').to(args.device)
    # else:
    #      model.load_state_dict(torch.load('/kaggle/input/pretrain-model-imagenette-final/resnet34_imagenette_final.pth', map_location=args.device))
    model.train()
    accum_steps = 8  # vì 256/40 ≈ 6.4 → làm tròn lên 7
    optim_g.zero_grad()
    # teacher_model.eval()
    # train_model_all_data(args,model,optim_model,trainloader, criterion, aug_rand)
    for batch_idx, (img_real, lab_real) in enumerate(trainloader):
        img_real = img_real.to(args.device)
        lab_real = lab_real.to(args.device)
        gen_loss = torch.tensor(0.0, requires_grad=True).to(args.device)
        tl= torch.tensor(0.0, requires_grad=True).to(args.device)
        loss = torch.tensor(0.0, requires_grad=True).to(args.device)

     
        # optim_g.zero_grad()

        noise = torch.randn(args.batch_size, args.dim_noise).to(args.device)
        lab_mapped = torch.tensor([IMAGENETTE_TO_IMAGENET[int(l)] for l in lab_real],
                              device=args.device)
        img_syn = generator(noise, lab_mapped)

        # 2. Update Student model to extract useful feature representations
        # train_match_model(args, model, optim_model, trainloader, criterion, aug_rand)

        # # 3. Match loss
        # if args.match_aug:
        #     img_aug = aug(torch.cat([img_real, img_syn]))
        #     match_loss = matchloss(args, img_aug[:args.batch_size], img_aug[args.batch_size:], lab_real, lab_real, model)
        # else:
        #     match_loss = matchloss(args, img_real, img_syn, lab_real, lab_real, model)

        # gen_loss +=match_loss *100
        
        # tesla_loss = loss2_tesla(args, model, img_syn, lab_real, teacher_model, T=5, beta=0.01)

        # gen_loss += 10 * tesla_loss
        # 4. Attention-based loss
        hooks = attach_hooks(activations, model)
        output_real,  logit_real= model(img_real)
        activations, original_model_set_activations = refreshActivations(activations)
        delete_hooks(hooks)

        hooks = attach_hooks(activations, model)
        output_syn, logit_syn = model(img_syn)
        activations, syn_model_set_activations = refreshActivations(activations)
        delete_hooks(hooks)

        length_of_network = len(original_model_set_activations)
        for layer in range(length_of_network - 1):
            # real_attention = attention_mixer(original_model_set_activations[layer].detach())[1]
            # syn_attention = attention_mixer(syn_model_set_activations[layer])[1]

            # real_attention = real_attention / (torch.sqrt(torch.var(real_attention, dim=0, keepdim=True) + 1e-8))*10
            # syn_attention = syn_attention / (torch.sqrt(torch.var(syn_attention, dim=0, keepdim=True) + 1e-8))*10
            
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

      




        
        pm_loss = prediction_matching_loss(logit_real.detach(), logit_syn, lab_real, temperature=2)
  
        gen_loss += 1000*pm_loss
        # gen_loss = loss1
        # loss_2 = loss2_class(logit_real.detach(), logit_syn,lab_real)
       
        # gen_loss +=  loss1+ 10*loss_2

      
        gen_loss.backward()
        if (batch_idx + 1) % accum_steps == 0:
            optim_g.step()
            optim_g.zero_grad()

        # 10. Logging
        gen_losses.update(gen_loss.item())
        # print(tabulate(table_data, headers=headers, tablefmt="grid"))
        if (batch_idx + 1) % args.print_freq == 0:
            print('[Train Epoch {} Iter {}] G Loss: {:.3f}({:.3f})'.format(
                epoch, batch_idx + 1, gen_losses.val, gen_losses.avg))
            # print(tabulate(table_data, headers=headers, tablefmt="grid"))

import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F


def train_match_model(args, model, optim_model, trainloader, criterion, aug_rand):
    '''The training function for the match model
    '''
    for batch_idx, (img, lab) in enumerate(trainloader):
        if batch_idx == args.epochs_match_train:
            break

        img = img.to(args.device)
        lab = lab.to(args.device)

        output = model(aug_rand(img))[1]
        loss = criterion(output, lab)

        optim_model.zero_grad()
        loss.backward()
        optim_model.step()

def train_model_all_data(args, model, optim_model, trainloader, criterion, aug_rand):
    '''The training function for the match model
    '''
    best_top1 = 0.0
    for epoch in range(300):
        model.train()
        for batch_idx, (img, lab) in enumerate(trainloader):
            img = img.to(args.device)
            lab = lab.to(args.device)
    
            output = model(aug_rand(img))[1]
            loss = criterion(output, lab)
    
            optim_model.zero_grad()
            loss.backward()
            optim_model.step()
        model.eval()
        test_top1, test_top5, test_loss = test(args, model, testloader, criterion)
        print('[Test Epoch {}] Top1: {:.3f} Top5: {:.3f}'.format(epoch, test_top1, test_top5))
        if test_top1 > best_top1:
            best_top1 = test_top1
            best_top5 = test_top5
            torch.save(
                     model.state_dict(),
                    os.path.join(args.output_dir, 'resnet_100.pth'.format(model)))
            print('Save model ')
    
 

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

import matplotlib.pyplot as plt
import torchvision.utils as vutils
def validate(args, generator, testloader, criterion, aug_rand):
    '''Validate the generator performance'''
    all_best_top1 = []
    all_best_top5 = []

    for e_model in args.eval_model:
        print('Evaluating {}'.format(e_model))
        model = define_model(args, args.num_classes, e_model).to(args.device)
        model.train()
        optim_model = torch.optim.SGD(model.parameters(), args.eval_lr, momentum=args.momentum,
                                      weight_decay=args.weight_decay)

        generator.eval()
        best_top1 = 0.0
        best_top5 = 0.0

        for epoch_idx in range(args.epochs_eval):
            total_samples = args.ipc * args.num_classes

            # --- Tạo 2 biến nhãn ---
            # Nhãn gốc 0..num_classes-1, dùng cho loss/accuracy
            lab_syn_full_orig = torch.cat([torch.full((args.ipc,), i, dtype=torch.long)
                                           for i in range(args.num_classes)]).to(args.device)

            # Nhãn map sang ImageNet1K, dùng cho generator
            lab_syn_full_imagenet = torch.tensor(
                [IMAGENETTE_TO_IMAGENET[int(l)] for l in lab_syn_full_orig],
                device=args.device
            )

            noise_full = torch.randn(total_samples, args.dim_noise).to(args.device)
            with torch.no_grad():
                img_syn_full = generator(noise_full, lab_syn_full_imagenet)
                img_syn_full = aug_rand((img_syn_full + 1.0) / 2.0)

            # Duyệt batch
            for batch_idx in range((total_samples + args.batch_size - 1) // args.batch_size):
                start_idx = batch_idx * args.batch_size
                end_idx = min(start_idx + args.batch_size, total_samples)
                img_syn = img_syn_full[start_idx:end_idx]
                lab_syn = lab_syn_full_orig[start_idx:end_idx]  # Dùng nhãn gốc cho loss/accuracy

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

                optim_model.zero_grad()
                loss.backward()
                optim_model.step()

            # Kiểm tra test
            if (epoch_idx + 1) % args.test_interval == 0:
                test_top1, test_top5, test_loss = test(args, model, testloader, criterion)
                print('[Test Epoch {}] Top1: {:.3f} Top5: {:.3f}'.format(epoch_idx + 1, test_top1, test_top5))
                if test_top1 > best_top1:
                    best_top1 = test_top1
                    best_top5 = test_top5

        all_best_top1.append(best_top1)
        all_best_top5.append(best_top5)

    return all_best_top1, all_best_top5


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

    trainloader, testloader = load_data(args)

    criterion = nn.CrossEntropyLoss()

    aug, aug_rand = diffaug(args)

    best_top1s = np.zeros((len(args.eval_model),))
    best_top5s = np.zeros((len(args.eval_model),))
    best_epochs = np.zeros((len(args.eval_model),))
    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        train(args, epoch, generator, discriminator, optim_g, optim_d, trainloader, criterion, aug, aug_rand)

        # save image for visualization
        generator.eval()
        test_label = torch.tensor(list(range(10)) * 5)
        test_label = torch.tensor([IMAGENETTE_TO_IMAGENET[int(l)] for l in test_label],
                            device=args.device)

        # test_noise = torch.normal(0, 1, (100, 100))
        # lab_onehot = torch.zeros((100, args.num_classes))
        # lab_onehot[torch.arange(100), test_label] = 1
        # test_noise[torch.arange(100), :args.num_classes] = lab_onehot[torch.arange(100)]
        test_noise  = torch.randn(50, args.dim_noise).cuda()
        test_noise = test_noise.cuda()
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
                                'discriminator': discriminator.state_dict(),
                                'optim_g': optim_g.state_dict(),
                                'optim_d': optim_d.state_dict()}
                    torch.save(
                        model_dict,
                        os.path.join(args.output_dir, 'model_dict_{}.pth'.format(e_model)))
                    print('Save model for {}'.format(e_model))

                print('Current Best Epoch for {}: {}, Top1: {:.3f}, Top5: {:.3f}'.format(e_model, best_epochs[e_idx], best_top1s[e_idx], best_top5s[e_idx]))




if __name__ == '__main__':
    main()