#!/usr/bin/env python
import sys
import math
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable, grad
from torchvision.utils import save_image
import os
slurm_name = os.environ["SLURM_JOB_ID"]
import argparse
from reg_loss import gan_loss
from LayerNorm1d import LayerNorm1d
import random

'''
Initially just implement LSGAN on MNIST.  

Then implement a critic.  
'''

parser = argparse.ArgumentParser(description='GANs and Adversarial Examples')
parser.add_argument('--use_adv_training', choices = (True, False), default=True)
parser.add_argument('--use_penalty', choices = (True,False), default=True)
parser.add_argument('--epsilon', type=float, default=0.03)
parser.add_argument('--epsilon_decay', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=10.0)
parser.add_argument('--adv_weight', type=float, default=1.0)
parser.add_argument('--attacktype', type=str, default="pgd")
parser.add_argument('--adv_iterations', type=float, default=20)
#parser.add_argument('--gan_type', type=str, default='roth', choices = ('roth', 'lsgan'))
#parser.add_argument('--model_levels', type=list, default=[4,32], choices = ([4],[4,32]))
#parser.add_argument('--compute_inception_score', type=bool, default=True)
#parser.add_argument('--num_epochs', type=int, default=3000)
#parser.add_argument('--batch_size', type=int, default=100)
#parser.add_argument('--dataname', type=str, default='cifar', choices = ('cifar', 'uniform', 'STL10', 'PTB_images'))
#parser.add_argument('--ae_4_4_model', type=str, default='saved_models/bottleneck_cifar_3_3.pt', choices=('saved_models/bottleneck_cifar_3_3.pt','/data/milatmp1/lambalex/ae_model_files/bottleneck_cifar_3_3.pt'))
#parser.add_argument('--save_images', type=bool, default=True)
#parser.add_argument('--render', action='store_true',
#                    help='render the environment')
#parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

print "args", args

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=True)

def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)

def gen_adv_example_fgsm(classifier, x, is_real,epsilon):
    #x_diff = 2 * 0.025 * (to_var(torch.rand(x.size())) - 0.5)
    #x_diff = torch.clamp(x_diff, -epsilon, epsilon)
    x_adv = x
    total_loss = 0.0
    
    c = classifier(x_adv)
    loss,_ = gan_loss(c, is_real,compute_penalty=False)
    nx_adv = x_adv + epsilon*torch.sign(grad(loss, x_adv,retain_graph=True)[0])
    x_adv = to_var(nx_adv.data)

    c = classifier(x_adv)
    loss,_ = gan_loss(c, is_real,compute_penalty=False)
    total_loss += loss

    return x_adv, total_loss

def gen_adv_example_pgd(classifier, x, is_real,iterations,step_size,epsilon):
    #x_diff = 2 * 0.025 * (to_var(torch.rand(x.size())) - 0.5)
    #x_diff = torch.clamp(x_diff, -epsilon, epsilon)
    #x_adv = x_diff + x
    x_adv = x
    total_loss = 0.0
    for i in range(0,iterations):
        c = classifier(x_adv)
        loss,_ = gan_loss(c, is_real,compute_penalty=False)
        nx_adv = x_adv + step_size*torch.sign(grad(loss, x_adv,retain_graph=True)[0])
        x_diff = nx_adv - x
        x_diff = torch.clamp(x_diff, -epsilon, epsilon)

        nx_adv = x_diff + x

        nx_adv = torch.clamp(nx_adv, -1.0, 1.0)
        x_adv = to_var(nx_adv.data)

    c = classifier(x_adv)
    loss,_ = gan_loss(c, is_real,compute_penalty=False)
    total_loss += loss


    return x_adv, total_loss

def gen_adv_example_basicgrad(classifier, x, is_real, iterations, step_size, epsilon):
    x_diff = 2 * 0.01 * (to_var(torch.rand(x.size())) - 0.5)
    x_diff = torch.clamp(x_diff, -epsilon, epsilon)
    x_adv = x_diff + x
    total_loss = 0.0
    for i in range(0,iterations):
        c = classifier(x_adv)
        loss,_ = gan_loss(c, is_real,compute_penalty=False)
        total_loss += loss
        g = grad(loss, x_adv,retain_graph=True)[0]
        g / g.norm(2)
        nx_adv = x_adv + 10000.0 * g
        x_diff = torch.clamp(nx_adv - x, -epsilon, epsilon)
        nx_adv = x_diff + x
        x_adv = to_var(nx_adv.data)

    c = classifier(x_adv)
    loss,_ = gan_loss(c, is_real,compute_penalty=False)
    total_loss += loss

    total_loss = total_loss * 0.1

    return x_adv, total_loss

def gen_adv_example(classifier, x, is_real, epsilon_use, attack_type="pgd"):
    if attack_type == "pgd":
        return gen_adv_example_pgd(classifier, x, is_real, args.adv_iterations, 0.01, epsilon_use)
    elif attack_type == "basicgrad":
        return gen_adv_example_basicgrad(classifier, x, is_real, args.adv_iterations, 0.01, epsilon_use)
    elif attack_type == "fgsm":
        return gen_adv_example_fgsm(classifier, x, is_real, epsilon_use)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])

#mnist = datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
mnist = datasets.STL10(root='/Tmp/lambalex/', split='train', download=True, transform=transform)
imglen = 96
imgchan = 3
inpdim = imglen*imglen*imgchan

data_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=100, shuffle=True)


#Discriminator
D = nn.Sequential(
    nn.Linear(inpdim, 1024),
    nn.LeakyReLU(0.02),
    nn.Linear(1024, 1024),
    nn.LeakyReLU(0.02),
    nn.Linear(1024,1024),
    nn.LeakyReLU(0.02),
    nn.Linear(1024, 1))

# Generator 
G = nn.Sequential(
    nn.Linear(64, 512),
    #nn.BatchNorm1d(512),
    nn.LeakyReLU(0.2),
    nn.Linear(512, 512),
    #nn.BatchNorm1d(512),
    nn.LeakyReLU(0.2),
    nn.Linear(512, inpdim),
    nn.Tanh())


if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()

d_optimizer = torch.optim.RMSprop(D.parameters(), lr=0.0001)
g_optimizer = torch.optim.RMSprop(G.parameters(), lr=0.0001)

for epoch in range(200):

    count = 0
    acc_clean = []
    acc_adv = []
    grad_penalty_lst = []

    epsilon_use = args.epsilon * math.exp(-args.epsilon_decay * epoch)

    print "epsilon current", epsilon_use

    for i, (images, _) in enumerate(data_loader):

        #count += 1
        #if count > 1:
        #    break

        batch_size = images.size(0)

        images = to_var(images.view(batch_size, -1))


        outputs = D(images)
        d_loss_real,grad_p_real = gan_loss(pre_sig=outputs, real=True, D=True, use_penalty=args.use_penalty,grad_inp=images,gamma=args.gamma)

        real_score = outputs

        z = to_var(torch.randn(batch_size, 64))
        fake_images = G(z)
        outputs = D(fake_images)

        d_loss_fake, grad_p_fake = gan_loss(pre_sig=outputs, real=False, D=True, use_penalty=args.use_penalty,grad_inp=fake_images,gamma=args.gamma)

        grad_penalty_lst.append(grad_p_real + grad_p_fake)

        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake

        D.zero_grad()
        d_loss.backward(retain_graph=True)
        d_optimizer.step()

        acc_clean.append(0.5*((fake_score < 0.0).type(torch.FloatTensor).mean() + (real_score > 0.0).type(torch.FloatTensor).mean()))

        x_fake_adv, loss_fake_adv = gen_adv_example(D, fake_images, False, epsilon_use, args.attacktype)
        x_real_adv, loss_real_adv = gen_adv_example(D, images, True, epsilon_use, args.attacktype)

        fake_score_adv = D(x_fake_adv)
        real_score_adv = D(x_real_adv)

        acc_adv.append(0.5*((fake_score_adv < 0.0).type(torch.FloatTensor).mean() + (real_score_adv > 0.0).type(torch.FloatTensor).mean()))

        fake_images_D = fake_images

        if args.use_adv_training:
            D.zero_grad()
            (args.adv_weight * (loss_fake_adv + loss_real_adv)).backward(retain_graph=False)
            d_optimizer.step()

        #print "fake scores D", fake_score[0:10]
        #print "real scores D", real_score[0:10]
        #print "epoch", epoch
        #============TRAIN GENERATOR========================$


        z = to_var(torch.randn(batch_size, 64))
        fake_images = G(z)

        outputs = D(fake_images)
     
        g_loss,_ = gan_loss(pre_sig=outputs, real=False, D=False, use_penalty=False,compute_penalty=False,grad_inp=None)

        D.zero_grad()
        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    print "Epoch", epoch
    print "Discriminator clean accuracy", sum(acc_clean)/len(acc_clean)
    print "Discriminator adv accuracy", sum(acc_adv)/len(acc_adv)
    print "Gradient Norm", sum(grad_penalty_lst)/len(grad_penalty_lst)

    fake_images = fake_images_D.view(fake_images_D.size(0), imgchan, imglen, imglen)
    save_image(denorm(fake_images.data), './data/%s_fake_images.png' %(slurm_name))

    fake_images_adv = x_fake_adv.view(x_fake_adv.size(0), imgchan, imglen, imglen)
    save_image(denorm(fake_images_adv.data), './data/%s_adv_fake_images.png' %(slurm_name))


