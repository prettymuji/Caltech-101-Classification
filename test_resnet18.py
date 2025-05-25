import torch
import torch.nn as nn
import utils
from torchvision.models import resnet18
import argparse

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
    
if __name__ == '__main__':
    
    # 创建解析器对象
    parser = argparse.ArgumentParser(description="程序描述")
    parser.add_argument("-p", "--pattern", help="初始化模式")  # 可选参数
    # 解析参数
    args = parser.parse_args()
    
    # 加载数据集
    _, _, test_loader = utils.data_prepare()
    # 加载模型
    model = resnet18()
    # 替换最后的全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 101)  # Caltech-101 有 101 个类别
    if args.pattern=='random':
        model.load_state_dict(torch.load('random_best_model.pth', weights_only=True))
    else:
        model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    
    # 测试
    test_model(model, test_loader)