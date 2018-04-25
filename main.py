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
import time
from inception_score import inception_score, IgnoreLabelDataset
import torch.utils.data as utils


'''
Initially just implement LSGAN on MNIST.  

Then implement a critic.  
'''

parser = argparse.ArgumentParser(description='GANs and Adversarial Examples')
parser.add_argument('--use_adv_training', choices = (True, False), default=False)
parser.add_argument('--use_penalty', choices = (True,False), default=False)
parser.add_argument('--epsilon', type=float, default=0.03)
parser.add_argument('--epsilon_decay', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--adv_weight', type=float, default=1.0)
parser.add_argument('--attacktype', type=str, default="fgsm")
parser.add_argument('--adv_iterations', type=float, default=10)
parser.add_argument('--step_size', type=float, default=0.01)
parser.add_argument('--train_gan_disc', default=True)
parser.add_argument('--train_classifier', default=False)
parser.add_argument('--train_classifier_adv', default=False)
parser.add_argument('--classifier_grad_penalty', default=False)
parser.add_argument('--compute_inception_score', default=True)
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

def to_var(x,requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

def denorm(x):
    #out = (x+1)/2
    out = x
    return out.clamp(0,1)

from attacks import gen_adv_example_fgsm, gen_adv_example_pgd

def gen_adv_example(classifier, x, loss_func, epsilon_use, attack_type="pgd"):
    if attack_type == "pgd":
        return gen_adv_example_pgd(classifier, x, loss_func, args.adv_iterations, args.step_size, epsilon_use)
    elif attack_type == "basicgrad":
        return gen_adv_example_basicgrad(classifier, x, loss_func, args.adv_iterations, 0.01, epsilon_use)
    elif attack_type == "fgsm":
        return gen_adv_example_fgsm(classifier, x, loss_func, epsilon_use)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.0,0.0,0.0), std=(1.0,1.0,1.0))])

#mnist = datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
#mnist_test = datasets.MNIST(root='./data/', train=False, download=True, transform=transform)

imglen = 32
imgchan = 3

cifar_train = datasets.CIFAR10('/Tmp/lambalex/', train=True, download=True,
                        transform=transforms.Compose([
                        transforms.CenterCrop(imglen),
                        transforms.ToTensor(),
                        transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                    ]))

cifar_test = datasets.CIFAR10('/Tmp/lambalex/', train=False, download=True,
                        transform=transforms.Compose([
                        transforms.CenterCrop(imglen),
                        transforms.ToTensor(),
                        transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                    ]))

imglen = 32
imgchan = 3

#mnist = datasets.STL10(root='/Tmp/lambalex/', split='train', download=True, transform=transform)
#imglen = 96
#imgchan = 3
inpdim = imglen*imglen*imgchan

data_loader = torch.utils.data.DataLoader(dataset=cifar_train, batch_size=100, shuffle=True)
data_loader_test = torch.utils.data.DataLoader(dataset=cifar_test, batch_size=100, shuffle=True)

nll = nn.CrossEntropyLoss()

from networks import Disc, Gen

from Classifier import Cl

#G = nn.Sequential(
#    nn.Linear(64, 512),
    #nn.BatchNorm1d(512),
#    nn.LeakyReLU(0.2),
#    nn.Linear(512, 512),
    #nn.BatchNorm1d(512),
#    nn.LeakyReLU(0.2),
#    nn.Linear(512, inpdim),
#    nn.Sigmoid())


D = Disc()
G = Gen()


if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()

#c_optimizer = torch.optim.Adam(C.parameters(), lr=0.0001, betas=(0.9,0.99))
d_optimizer = torch.optim.RMSprop(D.parameters(), lr=0.0001)
g_optimizer = torch.optim.RMSprop(G.parameters(), lr=0.0001)

for epoch in range(200):

    count = 0
    acc_clean = []
    acc_adv = []
    acc_class_clean = []
    acc_class_adv = []
    grad_penalty_lst = []
    gen_example_lst = []

    epsilon_use = args.epsilon * math.exp(-args.epsilon_decay * epoch)

    #print "epsilon current", epsilon_use

    for i, (images, true_labels) in enumerate(data_loader):

        print "iteration", i

        batch_size = images.size(0)

        #images = to_var(images.view(batch_size, -1))
        images = to_var(images)

        true_labels = to_var(true_labels,False)

        outputs, y_label_pred_real = D(images)
        d_loss_real,grad_p_real = gan_loss(pre_sig=outputs, real=True, D=True, use_penalty=args.use_penalty,grad_inp=images,gamma=args.gamma)
        
        class_loss, class_acc = D.compute_loss_acc(y_label_pred_real, true_labels)

        acc_class_clean.append(class_acc)
       
        image_class_adv,_ = gen_adv_example(D.label_classifier, images, lambda p: (nll(p, true_labels),None),args.epsilon)

        pred_labels = D.label_classifier(image_class_adv)

        class_loss_adv, class_acc_adv_curr = D.compute_loss_acc(pred_labels, true_labels)
        acc_class_adv.append(class_acc_adv_curr)

        real_score = outputs

        z = to_var(torch.randn(batch_size, 64))
        fake_images = G(z)
        gen_example_lst.append(fake_images.resize(fake_images.size(0),3,32,32).data)
        outputs,_ = D(fake_images)

        d_loss_fake, grad_p_fake = gan_loss(pre_sig=outputs, real=False, D=True, use_penalty=args.use_penalty,grad_inp=fake_images,gamma=args.gamma)

        grad_penalty_lst.append(grad_p_real.data + grad_p_fake.data)

        fake_score = outputs

        d_loss = 0.0
        if args.train_gan_disc:
            d_loss += d_loss_real + d_loss_fake
        if args.train_classifier:
            d_loss += class_loss
        if args.train_classifier_adv:
            d_loss += class_loss_adv
        if args.classifier_grad_penalty:
            d_loss += 10.0 * grad(y_label_pred_real.mean(), images,create_graph=True)[0].norm(2)


        D.zero_grad()
        d_loss.backward(retain_graph=True)
        d_optimizer.step()

        acc_clean.append((0.5*((fake_score < 0.0).type(torch.FloatTensor).mean() + (real_score > 0.0).type(torch.FloatTensor).mean())).data)

        x_fake_adv, loss_fake_adv = gen_adv_example(D.discriminator, fake_images, lambda cval: gan_loss(cval, real=False,compute_penalty=False), epsilon_use, args.attacktype)
        x_real_adv, loss_real_adv = gen_adv_example(D.discriminator, images, lambda cval: gan_loss(cval, real=True,compute_penalty=False), epsilon_use, args.attacktype)

        fake_score_adv,_ = D(x_fake_adv)
        real_score_adv,_ = D(x_real_adv)

        acc_adv.append((0.5*((fake_score_adv < 0.0).type(torch.FloatTensor).mean() + (real_score_adv > 0.0).type(torch.FloatTensor).mean())).data)

        fake_images_D = fake_images

        if args.use_adv_training:
            D.zero_grad()
            (args.adv_weight * (loss_fake_adv + loss_real_adv)).backward(retain_graph=False)
            d_optimizer.step()

        #============TRAIN GENERATOR========================$

        z = to_var(torch.randn(batch_size, 64))
        fake_images = G(z)

        outputs,_ = D(fake_images)
     
        g_loss,_ = gan_loss(pre_sig=outputs, real=False, D=False, use_penalty=False,compute_penalty=False,grad_inp=None)

        D.zero_grad()
        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    acc_class_clean_test = []
    acc_class_adv_test = []
    grad_penalty_lst = []

    #print "epsilon current", epsilon_use

    for i, (images, true_labels) in enumerate(data_loader_test):
        batch_size = images.size(0)
        images = to_var(images)
        true_labels = to_var(true_labels,False)
        outputs, y_label_pred_real = D(images)
        class_loss, class_acc = D.compute_loss_acc(y_label_pred_real, true_labels)
        acc_class_clean_test.append(class_acc)
        image_class_adv,_ = gen_adv_example(D.label_classifier, images, lambda p: (nll(p, true_labels),None),args.epsilon)
        pred_labels = D.label_classifier(image_class_adv)
        class_loss_adv, class_acc_adv_curr = D.compute_loss_acc(pred_labels, true_labels)
        acc_class_adv_test.append(class_acc_adv_curr)

    print "Epoch", epoch
    if args.train_gan_disc:
        print "Discriminator clean accuracy", sum(acc_clean)/len(acc_clean)
        print "Discriminator adv accuracy", sum(acc_adv)/len(acc_adv)
    print "Classifier Train clean accuracy", sum(acc_class_clean) / len(acc_class_clean)
    print "Classifier Train adv accuracy", sum(acc_class_adv) / len(acc_class_adv)
    print "Classifier Test clean accuracy", sum(acc_class_clean_test) / len(acc_class_clean_test)
    print "Classifier Test adv accuracy", sum(acc_class_adv_test) / len(acc_class_adv_test)    

#print "Gradient Norm", sum(grad_penalty_lst)/len(grad_penalty_lst)


    fake_images = fake_images_D.view(fake_images_D.size(0), imgchan, imglen, imglen)
    save_image(denorm(fake_images.data), './data/%s_fake_images.png' %(slurm_name))

    fake_images_adv = x_fake_adv.view(x_fake_adv.size(0), imgchan, imglen, imglen)
    save_image(denorm(fake_images_adv.data), './data/%s_adv_fake_images.png' %(slurm_name))

    if args.compute_inception_score and epoch % 5 == 1:
        t0 = time.time()
        genex_stack = torch.cat(gen_example_lst)
        gen_ex_ds = IgnoreLabelDataset(utils.TensorDataset(genex_stack,genex_stack))
        print "Inception Score", inception_score(gen_ex_ds, resize=True)
        print "Time to compute inception score", time.time() - t0
        del gen_ex_ds
        del gen_example_lst
        del genex_stack









