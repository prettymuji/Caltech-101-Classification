# 微调预训练ResNet-18模型实现Caltech-101分类
## 项目简介
该项目微调在ImageNet上预训练的卷积神经网络ResNet-18实现Caltech-101分类。实验发现，使用预训练参数微调的训练方式训练分类器，测试集上的分类准确率将达到96.59，与从随机初始化的网络参数开始训练得到的结果相比，高了约20%准确率。
## 数据集
本项目所使用的数据集来自官网 https://data.caltech.edu/records/mzrjq-6wc02，在训练开始前，请前往 Caltech-101 数据集官网选择‘caltech-101.zip’下载压缩文件至当前目录下并命名为‘caltech-101’，对其子文件‘101\_ObjectCategories.tar.gz’进行解压。本次实验只需要用到‘./caltech-101/101\_ObjectCategories’中的数据。
## 仓库结构
```
├── contrast_test.py                  # 随机初始化模型的训练框架
├── train_resnet18.py                 # 预训练模型的训练框架
├── params_search.py                  # 参数查找
├── test_resnet18.py                  # 测试最优参数在测试集上的准确率
├── utils.py                          # 数据加载与预处理、模型参数初始化、参数查找结果输出
├── README.md                         # 项目说明文件
├── runs                              # Tensorboard可视化结果
├── hyper_search.csv                  # 参数查找结果
```
## 模型训练
- 直接在命令行运行
<pre><code>python train_resnet18.py</code></pre>
即可使用最优超参数组合微调ImageNet预训练模型实现Caltech-101分类；
- 直接在命令行运行
<pre><code>python contrast_test.py</code></pre>
即可从随机初始化网络参数开始训练模型ResNet-18模型实现分类。
## 测试
如需测试模型，请先前往 http://dsw-gateway.cfff.fudan.edu.cn:32080/dsw-15320/lab/tree/zl_dl/hw2-1/best_model.pth 下载预训练微调模型的最优参数于当前文件夹下并命名为`best_model.pth`，前往 http://dsw-gateway.cfff.fudan.edu.cn:32080/dsw-15320/lab/tree/zl_dl/hw2-1/random_best_model.pth 下载随机初始化参数的训练模型于当前文件夹下并命名为`random_best_model.pth`。测试方式如下：
- 直接在命令行运行
<pre><code>python test_resnet18.py</code></pre>
即可输出最优参数组合的最优模型 `best_model.pth` 在测试集上的准确率；
- 直接在命令行运行
<pre><code>python test_resnet18.py -p random</code></pre>
即可输出最优参数组合的最优模型 `random_best_model.pth` 在测试集上的准确率。
## 参数查找
- 直接在命令行运行
<pre><code>python params_search.py</code></pre>
运行结束后会在该目录下创建新的文件夹 "params_new" , 不同超参数对应的收敛信息及相应的模型参数会保存在对应子文件夹下；
- hyper_search.csv 存储了实验所涉及的超参数组合训练结果，可以直接查看；
- 参数查找环节的所有训练好的模型权重存储在 http://dsw-gateway.cfff.fudan.edu.cn:32080/dsw-15320/lab/tree/zl_dl/hw2-1/params_new 文件夹对应的子文件夹下。
## 模型权重下载
- 预训练最优模型参数在 `best_params.pth` 中, 运行 `test_resnet18.py` 即可加载该模型并输出在测试集上的准确率；
- 随机初始化最优模型参数在 `random_best_params.pth` 中, 运行 `test_resnet18.py -p random` 即可加载该模型并输出在测试集上的准确率；
- 参数查找环节的所有参数位于 http://dsw-gateway.cfff.fudan.edu.cn:32080/dsw-15320/lab/tree/zl_dl/hw2-1/params_new 可以直接下载整个文件夹于当前目录, 运行`utils.hyper_search()`函数即可依次输出所有超参数组合训练好的模型在测试集上的准确率、最佳准确率对应的超参数组合，相应的结果会储存在'hyper_search.csv'文件中。
