import torch
from reg_loss import gan_loss
from torch.autograd import grad, Variable

def to_var(x,requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def gen_adv_example_fgsm(classifier, x, loss_func,epsilon):
    #x_diff = 2 * 0.025 * (to_var(torch.rand(x.size())) - 0.5)
    #x_diff = torch.clamp(x_diff, -epsilon, epsilon)
    x_adv = x
    total_loss = 0.0

    c_pre = classifier(x_adv)
    loss,_ = loss_func(c_pre) # gan_loss(c, is_real,compute_penalty=False)
    nx_adv = x_adv + epsilon*torch.sign(grad(loss, x_adv,retain_graph=True)[0])
    x_adv = to_var(nx_adv.data)

    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    c = classifier(x_adv)
    loss,_ = loss_func(c)#gan_loss(c, is_real,compute_penalty=False)
    
    total_loss = ((c - c_pre)**2).mean()

    return x_adv, total_loss

def gen_adv_example_pgd(classifier, x, loss_func,iterations,step_size,epsilon):
    x_diff = 2 * 0.025 * (to_var(torch.rand(x.size())) - 0.5)
    x_diff = torch.clamp(x_diff, -epsilon, epsilon)
    x_adv = x_diff + x
    x_adv = x
    total_loss = 0.0
    for i in range(0,iterations):
        c = classifier(x_adv)
        loss,_ = loss_func(c)
        nx_adv = x_adv + step_size*torch.sign(grad(loss, x_adv,retain_graph=True)[0])
        x_diff = nx_adv - x
        x_diff = torch.clamp(x_diff, -epsilon, epsilon)
        nx_adv = x_diff + x
        nx_adv = torch.clamp(nx_adv, -1.0, 1.0)
        x_adv = to_var(nx_adv.data)

    c = classifier(x_adv)
    loss,_ = loss_func(c)
    total_loss += loss


    return x_adv, total_loss


