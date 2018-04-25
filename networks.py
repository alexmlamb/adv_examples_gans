import torch
import torch.nn as nn

nll = nn.CrossEntropyLoss()

class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()

        self.l0 = nn.Sequential(nn.Linear(64, 512*4*4))

        self.l1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.02),
            nn.Conv2d(32, 3, kernel_size=5, padding=2, stride=1),
            nn.Sigmoid())

    def forward(self,x):
        h = self.l0(x)
        h = h.resize(h.size(0),512, 4, 4)
        h = self.l1(h)
        #h = h.resize(h.size(0),3*32*32)
        return h

class Disc(nn.Module):

    def __init__(self):
        super(Disc, self).__init__()

        self.D1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(0.02),
            nn.Conv2d(128, 128, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.02),
            nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(0.02),
            nn.Conv2d(256, 256, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.02),
            nn.Conv2d(256, 512, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.02))

        self.D_labels = nn.Sequential(
            nn.Linear(512*4*4,10))

        self.D_disc = nn.Linear(512*4*4,1)
        self.nll_loss = nll

    def compute_h(self,x):
        #x = x.resize(x.size(0), 3, 32, 32)

        h = self.D1(x)

        h = h.resize(x.size(0), 512*4*4)

        return h

    def forward(self,x):

        h = self.compute_h(x)

        y_labels = self.D_labels(h)
        y_disc = self.D_disc(h)

        return y_disc, y_labels

    def discriminator(self,x):
        h = self.compute_h(x)
        return self.D_disc(h)

    def label_classifier(self,x):
        h = self.compute_h(x)
        y_labels = self.D_labels(h)
        return y_labels

    def compute_loss_acc(self,y_labels, true_labels):

        loss = nll(y_labels, true_labels)

        _, label_pred = torch.max(y_labels.data,1)

        acc = label_pred.eq(true_labels.data.view_as(label_pred)).cpu().double().mean()

        return loss, acc


