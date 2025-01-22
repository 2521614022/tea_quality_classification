import os
import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
from d2l import torch as d2l
from tqdm import tqdm

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid(visible=False)

class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                ylim=None, xscale='linear',yscale='linear',
                fmts=('#ffd323', '#88aa29', '#ef3054'),nrows=1,ncols=1,
                figsize=(10,6)): 
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows,ncols,figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes,]
        self.config_axes = lambda: set_axes(self.axes[0],xlabel,ylabel,xlim,ylim,xscale,yscale,legend)         
        self.X, self.Y, self.fmts = None, None, fmts
        
    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)] 
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a,b) in enumerate(zip(x,y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt, linewidth=5)
        self.config_axes()

# Define a function that uses multi-GPU patterns for training and evaluation
def train_batch(net, X, y, loss, trainer, devices):
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X] # If X is a list, move the data one by one to devices [0]    
    else:
        X = X.to(devices[0]) # If X is not a list, move X to devices [0]
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train(net, train_iter, test_iter, loss, trainer, num_epochs, lr_period, lr_decay, devices=d2l.try_all_gpus()):
    scheduler = torch.optim.lr_scheduler.StepLR(trainer,lr_period,lr_decay)
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator_acc = Animator(xlabel='epoch',xlim=[1,num_epochs],ylim=[0,1.05],
                           legend=['train acc', 'test acc'])
    animator_loss = Animator(xlabel='epoch',xlim=[1,num_epochs],ylim=[0,6.00],
                            legend=['train loss'])
    # Working with multiple GPUs
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        with tqdm(total=num_batches, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            metric = d2l.Accumulator(4)
            for i, (features, labels) in enumerate(train_iter):
                timer.start()
                l, acc = train_batch(net,features,labels,loss,trainer,devices) 
                metric.add(l,acc,labels.shape[0],labels.numel())
                timer.stop()
                if (i + 1) % (num_batches // 5) == 0 or i == num_batches -1:
                    animator_acc.add(
                        epoch + (i + 1) / num_batches,
                        (metric[1] / metric[3], None)) 
                    animator_loss.add(
                        epoch + (i + 1) / num_batches,
                        (metric[0] / metric[2]))
                pbar.update()
                pbar.set_postfix_str(f'Loss: {l.item()/labels.shape[0]:.4f}')
        test_acc = d2l.evaluate_accuracy_gpu(net,test_iter)
        animator_acc.add(epoch+1,(None,test_acc))
        scheduler.step()
    print(f'loss {metric[0] / metric[2]:.3f}, train acc'
         f' {metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f' {metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
         f' {str(devices)}')  

def train_fine_tuning(net, learning_rate, train_augs, test_augs, lr_period, lr_decay, batch_size, num_epochs, param_group=True):
    train_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join('../data', 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True
    )
    test_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join('../data', 'test'), transform=test_augs),
        batch_size=batch_size
    )
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction='none')
    if param_group:
        # The default learning rate is used except for the learning rate of the last layer
        # The learning rate of the last layer is ten times the learning rate
        params_lx = [
            param for name, param in net.named_parameters()
            if name not in ["fc.weight","fc.bias"] ]
        trainer = torch.optim.SGD([
            {'params': params_lx}, 
            {'params': net.fc.parameters(), 'lr': learning_rate * 10}],
            lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(),lr=learning_rate,weight_decay=0.001)
    train(net, train_iter, test_iter, loss, trainer, num_epochs, lr_period, lr_decay, devices)   

def main():
    train_imgs = torchvision.datasets.ImageFolder(os.path.join('../data', 'train'))
    test_imgs = torchvision.datasets.ImageFolder(os.path.join('../data', 'test'))
    
    # Data augmentation
    normalize = torchvision.transforms.Normalize([0.485,0.456,0.406],
                                                 [0.229,0.224,0.225])
    train_augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        normalize
    ])
    test_augs = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        normalize
    ])
    
    # Define and initialize models
    finetune_net = torchvision.models.resnet18(pretrained=True)
    finetune_net.fc = nn.Linear(finetune_net.fc.in_features,2)
    nn.init.xavier_uniform_(finetune_net.fc.weight)
    
    # Use a smaller learning rate
    lr = 5e-5
    batch_size = 128
    num_epochs = 10
    lr_period = num_epochs // 5
    lr_decay = 0.9
    
    train_fine_tuning(finetune_net, lr, train_augs, test_augs, lr_period, lr_decay, batch_size, num_epochs)
    # torch.save(finetune_net.state_dict(), '../models/finetune_resnet_straightness.pth')
    # torch.save(finetune_net.state_dict(), '../models/finetune_resnet_smoothness.pth')
    # torch.save(finetune_net.state_dict(), '../models/finetune_resnet_tenderness.pth')
    # torch.save(finetune_net.state_dict(), '../models/finetune_resnet_moisture.pth')
    # torch.save(finetune_net.state_dict(), '../models/finetune_resnet_fragmentation.pth')
    # torch.save(finetune_net.state_dict(), '../models/finetune_resnet_greenness.pth')
    torch.save(finetune_net.state_dict(), '../models/finetune_resnet_flatness.pth')
    # torch.save(finetune_net.state_dict(), '../models/finetune_resnet_uniformity.pth')
        
    plt.show()
    
if __name__ == '__main__':
    main()