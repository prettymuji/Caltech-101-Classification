from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset 
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
# import test_resnet18
import os
import torch

def data_prepare(batchsize=32, verbose=True):
    # 定义数据增强与预处理
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]), # ImageNet的标准化
    ])

    # 加载整个数据集
    full_dataset = datasets.ImageFolder(root = './caltech-101/101_ObjectCategories', transform=transform)
    
    # 构建索引列表
    train_idx, val_idx, test_idx = [], [], []
    # 按照8:1:1划分数据集
    for cls_idx in range(len(full_dataset.classes)):
        inds = [i for i,(path,label) in enumerate(full_dataset.samples) if label==cls_idx]
        train_end_id = int(np.ceil(len(inds)*0.8))
        train_idx += inds[:train_end_id]
        val_end_id = int(np.ceil(len(inds)*0.9))
        val_idx   += inds[train_end_id: val_end_id]
        test_idx  += inds[val_end_id:]
    
    # 合成数据集
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)
        
    # 定义 DataLoader
    # 这里的dataloader每次都会进行采样，所以会耗费cpu，很影响效率，之后优化要改写一下
    # train_loader = DataLoader(train_dataset, batch_size=batchsize,
    #                           sampler=SubsetRandomSampler(train_idx), num_workers=16, pin_memory=True, prefetch_factor=4) ## 进程太慢啦！！要加pin_memory 和 num_workers
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=4, persistent_workers=True)
    val_loader   = DataLoader(val_dataset, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=4, persistent_workers=True)
    test_loader  = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=4, persistent_workers=True)
    
    # 输出信息
    if verbose:
        print(f'训练集: {len(train_idx)}, 验证集: {len(val_idx)}, 测试集: {len(test_idx)}.')
    
    return train_loader, val_loader, test_loader

# 初始化resnet18
def init_resnet18(pretrain=True):
    if pretrain:
        # 加载预训练的 ResNet-18 模型
        weights = ResNet18_Weights.DEFAULT  # 获取默认的预训练权重
        model = resnet18(weights=weights)
    else:
        model = resnet18()
    
    # 替换最后的全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 101)  # Caltech-101 有 101 个类别
    
    return model

# 测试模型(smwy, hyper_search直接调test_resnet18文件里面的函数永远调的是旧版的，怎么保存怎么刷新都不对，鬼知道怎么了)
def test_model(model, test_loader, verboses=True):
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = correct / total
    if verboses:
        print(f"Test Accuracy: {test_acc:.4f}")
    return test_acc

# 从文件名获取超参数
def get_hyperparams_from_file(path):
    # path = './bs=32, lr_fc=0.001, lr_others=0.0001, num_epoch=10, scheduler_flag=False'
    info = path.split('/')[-1]
    info = info.split(',')
    bs = int(info[0].split('=')[-1])
    lr_fc = float(info[1].split('=')[-1])
    lr_others = float(info[2].split('=')[-1])
    num_epoch = int(info[3].split('=')[-1])
    scheduler_flag = info[4].split('=')[-1]
    return bs, lr_fc, lr_others, num_epoch, scheduler_flag

# 整理超参数搜索结果
def hyper_search(path = './params_new'):
    # 统计文件个数
    total = {'batch size':[], 'lr of fc':[], 'lr of other layers':[], 'epochs':[], 'scheduler':[], 'acc on test dataset':[]}
    file_name = [file for file in os.listdir(path) if file != '.ipynb_checkpoints']
    best_acc = 0.0
    best_params = {}
    for file_dir in file_name:
        file_dir = os.path.join(path, file_dir)
        # 判空
        if len(os.listdir(file_dir)) == 0:
            continue

        params_dir = os.path.join(file_dir, 'best_model.pth')
        bs, lr_fc, lr_others, num_epoch, scheduler_flag = get_hyperparams_from_file(os.path.join(file_dir))

        total['batch size'].append(bs)
        total['lr of fc'].append(lr_fc)
        total['lr of other layers'].append(lr_others)
        total['epochs'].append(num_epoch)
        total['scheduler'].append(scheduler_flag)

        # 复原模型
        # 加载数据集
        _, _, test_loader = data_prepare(verbose=False)
        # 加载模型
        model = resnet18()
        # 替换最后的全连接层
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 101)  # Caltech-101 有 101 个类别
        model.load_state_dict(torch.load(params_dir, weights_only=True))
    
        # 测试
        test_acc = test_model(model, test_loader, verboses=False)
        
        total['acc on test dataset'].append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            best_params = dict(bs=bs, lr_fc=lr_fc, lr_others=lr_others, num_epoch=num_epoch, scheduler_flag=scheduler_flag)
        print(f'超参数组合: batch_size={bs}, lr_fc={lr_fc}, lr_others={lr_others}, epochs={num_epoch}, scheduler={scheduler_flag}| 测试集acc: {test_acc:4f}')
    
    print('-'*20)
    print(f'最优超参数组合: \n{best_params}')
    print(f'最优正确率: {best_acc}')
    return total
        