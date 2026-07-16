import contextlib
import logging
import os
import glob

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from utils import *
from SeqRec.sasrec.model import SASRec
# from Seq_Exp.SeqRec.sasrec.model import SASRec

#这段代码是 LLM-SRec 项目中“连接层”的一部分。
#它的核心作用是：加载第一阶段训练好的 SASRec 模型
#并将其“冻结” (Freeze)，使其成为一个只负责输出特征、不参与后续梯度更新的 Teacher 模型。

#负责从硬盘找到并读取模型文件
def load_checkpoint(recsys, pre_trained):
    
    # 1. 拼凑路径
    # 比如 ./SeqRec/SASRec/Beauty/
    path = f'./SeqRec/{recsys}/{pre_trained}/'
    # 2. 寻找 .pth 文件
    # find_filepath 是一个工具函数，去目录里找后缀是 .pth 的文件
    pth_file_path = find_filepath(path, '.pth')
    # 3. 安全检查
    # 必须只能找到 1 个模型文件，如果有多个它不知道加载哪个，会报错
    assert len(pth_file_path) == 1, 'There are more than two models in this dir. You need to remove other model files.\n'
    # 4. 加载 (Resurrection)
    # torch.load 读出来的就是我们上一节分析的那个列表：[kwargs, checkpoint]
    # kwargs: 模型的配置参数 (Hidden Size, Layer Num...)
    # checkpoint: 具体的权重参数
    kwargs, checkpoint = torch.load(pth_file_path[0], map_location="cpu", weights_only= False)
    logging.info("load checkpoint from %s" % pth_file_path[0])

    return kwargs, checkpoint

#这是 LLM-SRec 用来管理 SASRec 的容器。
class RecSys(nn.Module):
    def __init__(self, recsys_model, pre_trained_data, device):
        super().__init__()
        # 1. 加载配置和权重
        kwargs, checkpoint = load_checkpoint(recsys_model, pre_trained_data)
        kwargs['args'].device = device
        # 2. 【复活】重建 SASRec 模型
        # 使用 **kwargs 自动填入参数，这就不用手动写 hidden_units=64 了
        model = SASRec(**kwargs)
        # 3. 【注入灵魂】加载权重
        model.load_state_dict(checkpoint)
        # 4. 【核心步骤】冻结参数 (Freezing)
        # 遍历 SASRec 的每一个参数 (w)，设置 requires_grad = False    
        for p in model.parameters():
            p.requires_grad = False
        
        # 5. 保存属性供 LLM 使用    
        self.item_num = model.item_num
        self.user_num = model.user_num

        self.model = model.to(device) # 把模型搬到 GPU 上
        self.hidden_units = kwargs['args'].hidden_units # 记录维度，比如 64
        
    def forward():
        print('forward')

'''这段代码最重要的意义
请注意第 4 步的 p.requires_grad = False。
这是 LLM-SRec (Distillation) 架构的关键：
Teacher 不再学习：我们在第二阶段训练时，只训练 Projector (MLP) 和 LLM (如果 LLM 没冻结的话)。
稳定特征源：SASRec 在这里就像一个**“特征提取器”**。对于同一个用户序列，它永远输出相同的 Embedding，不会因为 LLM 的训练而发生抖动或改变。
节省显存：因为不需要计算 SASRec 的梯度，训练开销会变小。
''' 