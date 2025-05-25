import utils
from train_resnet18 import train_model
import itertools
import torch
from torch import nn, optim
import os
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    # batch_size, lr, num_epoch
    bs = [32, 64]
    lr_fc = [1e-2, 1e-3]
    lr_others = [1e-3, 3e-4, 1e-4, 0]
    num_epoch = [10, 20, 30]
    scheduler_flag = [True, False]
    
    for bs, lr_fc, lr_others, num_epoch, scheduler_flag \
    in itertools.product(bs, lr_fc, lr_others, num_epoch, scheduler_flag): # 笛卡尔积
            
        # 加载数据
        batch_size = bs
        train_loader, val_loader, _ = utils.data_prepare(batchsize=batch_size)
        
        # 初始化模型
        model = utils.init_resnet18()

        # 将模型移动到 GPU（如果可用）
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'device = {device}')

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        
        # 定义优化器
        weight_decay = 1e-4
        optimizer = optim.SGD([
            {'params': model.fc.parameters(), 'lr': lr_fc},
            {'params': [param for name, param in model.named_parameters() if name not in ['fc.weight', 'fc.bias']], 'lr': lr_others}], momentum=0.9, weight_decay=weight_decay) #  weight_decay为L2衰减
        
        # 移动模型
        model = model.to(device)
        
        # 设置学习率调度器
        if scheduler_flag:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        else:
            scheduler = None
        
        # 训练模型
        folder = f'./params_new/bs={bs}, lr_fc={lr_fc}, lr_others={lr_others}, num_epoch={num_epoch}, scheduler_flag={scheduler_flag}'
        print(f' bs={bs},\n lr_fc={lr_fc},\n lr_others={lr_others},\n num_epoch={num_epoch},\n scheduler_flag={scheduler_flag}:')
        if os.path.exists(folder):
            print(f'{folder}已存在')
            continue
            os.makedirs(folder, exist_ok=True)
        
        log_dir = os.path.join(folder, 'runs')
        writer = SummaryWriter(log_dir=log_dir)
        path = os.path.join(folder, 'best_model.pth')
        train_model(model, train_loader, val_loader,criterion, optimizer, device, writer, scheduler=scheduler, path = path, num_epochs=num_epoch)
        writer.close()

        