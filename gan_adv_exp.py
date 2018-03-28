#!/usr/bin/env python
import sys
sys.path.insert(0, '/u/lambalex/.local/lib/python2.7/site-packages/torch-0.2.0+4af66c4-py2.7-linux-x86_64.egg')
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

from reg_loss import gan_loss
from utils import to_var
from torch.autograd import grad
from torch.nn.modules import NLLLoss
from attacks import gen_adv_example_fgsm
from Classifier import Cl

C = Cl()

c_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, C.parameters()), lr=0.0001, betas=(0.9,0.99))

nll = nn.CrossEntropyLoss()

def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.0,0.0,0.0), std=(1.0,1.0,1.0))])

mnist_train = datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data/', train=False, download=True, transform=transform)

data_loader_train = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=100, shuffle=True)
data_loader_test = torch.utils.data.DataLoader(dataset=mnist_test, batch_size=100, shuffle=True)

if torch.cuda.is_available():
    C = C.cuda()

for epoch in range(0,50):
    for i, (images, labels) in enumerate(data_loader_train):

        batch_size = images.size(0)

        images = to_var(images.view(batch_size, -1))

        labels = to_var(labels,requires_grad=False)

        pred = C(images)

        nll_loss = nll(pred, labels).mean()

        grad_penalty = grad(pred.mean(), images, create_graph=True)[0].norm(2).mean() * 0.0

        loss = nll_loss + grad_penalty

        _, label_pred = torch.max(pred.data, 1)

        acc = label_pred.eq(labels.data.view_as(label_pred)).cpu().double().mean()

        C.zero_grad()
        loss.backward(retain_graph=True)
        c_optimizer.step()


    test_acc = []
    test_acc_fgsm = []
    for i, (images, labels) in enumerate(data_loader_train):
        images = to_var(images.view(batch_size, -1))
        labels = to_var(labels,requires_grad=False)
        pred = C(images)
        nll_loss = nll(pred, labels).mean()
        _, label_pred = torch.max(pred.data, 1)
        acc = label_pred.eq(labels.data.view_as(label_pred)).cpu().double().mean()
        test_acc.append(acc)

        x_fgsm,_ = gen_adv_example_fgsm(C, images, lambda p: (nll(p, labels),None),0.03)
        #x_grad = grad(nll_loss, images)[0]
        #x_fgsm = images + torch.sign(x_grad)*0.3
        pred = C(x_fgsm)
        _, label_pred = torch.max(pred.data, 1)
        acc = label_pred.eq(labels.data.view_as(label_pred)).cpu().double().mean()
        test_acc_fgsm.append(acc)

    print "Test Acc", epoch, sum(test_acc)/len(test_acc)
    print "FGSM Acc", epoch, sum(test_acc_fgsm)/len(test_acc_fgsm)











