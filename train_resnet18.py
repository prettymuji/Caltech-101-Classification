import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18, ResNet18_Weights
import utils

# 训练和验证函数
def train_model(model, train_loader, val_loader, criterion, optimizer, device, writer, scheduler=None, path = 'best_model.pth', num_epochs=25):
    '''
    model: 载入的模型
    train_loader, val_loader: 数据
    criterion: 损失函数
    optimizer: 优化器
    scheduler: LR衰减模式
    device
    writer: tensorboard日志
    path: 最优参数存储路径
    '''
    
    best_acc = 0.0
    # criterion = crite.to(device)  # 在训练前定义
    # print(criterion)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 每个 epoch 包含训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total = 0

            # 遍历数据
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # loss = nn.CrossEntropyLoss()(outputs, labels)

                    # 反向传播和优化
                    if phase == 'train':
                        optimizer.zero_grad()
                        # print("inputs on:", inputs.device)
                        # print("labels on:", labels.device)
                        # print("models on:", next(model.parameters()).device)
                        
                        loss.backward()
                        optimizer.step()

                # 统计损失和准确率
                total += labels.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.double() / total

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 将结果写入 TensorBoard
            # if phase == 'train':
            #     writer.add_scalar('Loss/train', epoch_loss, epoch)
            #     writer.add_scalar('Accuracy/train', epoch_acc, epoch)
            # else:
            #     writer.add_scalar('Loss/val', epoch_loss, epoch)
            #     writer.add_scalar('Accuracy/val', epoch_acc, epoch)
            if phase == 'train':
                train_loss = epoch_loss
                train_acc = epoch_acc
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc
            

            # 保存最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), path)
                
        # 将结果写入 TensorBoard
        writer.add_scalars('Loss',
                           {'train': train_loss,            # 子曲线名 'train'
                            'val':   val_loss},             # 子曲线名 'val'
                           epoch)                            # x 轴坐标
        writer.add_scalars('Accuracy',
                           {'train': train_acc,            # 子曲线名 'train'
                            'val':   val_acc},             # 子曲线名 'val'
                           epoch)                            # x 轴坐标

        if scheduler is not None:
            scheduler.step()

    print(f'Best val Acc: {best_acc:.4f}')


if __name__ == '__main__':
    
    # 加载数据
    batch_size = 32
    train_loader, val_loader, _ = utils.data_prepare(batchsize=batch_size)
    
    # 初始化模型
    model = utils.init_resnet18()
    
    # 将模型移动到 GPU（如果可用）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device = {device}')
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    # 定义优化器
    lr_fc = 1e-2
    lr_others = 1e-3
    weight_decay = 5e-4
    optimizer = optim.SGD([
        {'params': model.fc.parameters(), 'lr': lr_fc},
        {'params': [param for name, param in model.named_parameters() if name not in ['fc.weight', 'fc.bias']], 'lr': lr_others}], momentum=0.9, weight_decay=weight_decay) #  weight_decay为L2衰减
    
    # 加载模型
    model = model.to(device)
    
    # 设置学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 训练模型
    writer = SummaryWriter(log_dir='runs/caltech101_resnet18')
    train_model(model, train_loader, val_loader,criterion, optimizer, device, writer, scheduler=scheduler, path = 'best_model.pth', num_epochs=30)
    writer.close()
    