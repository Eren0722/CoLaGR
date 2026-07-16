import os
import time
import torch
import argparse
import numpy as np
import sys

from model import SASRec
from data_preprocess import *
from utils import *

from tqdm import tqdm
#这是 SASRec 模型的主训练脚本 (main.py)。
#1. 参数定义 (argparse)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=128, type=int)
parser.add_argument('--hidden_units', default=64, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.1, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='0', type=str, help='cpu, hpu, gpu -> num')

parser.add_argument('--inference_only', default=False, action='store_true') #如果加上这个参数，就跳过训练，直接加载模型进行测试。
parser.add_argument('--nn_parameter', default=False, action='store_true')
parser.add_argument('--state_dict_path', default=None, type=str)

args = parser.parse_args()

if __name__ == '__main__':
    
    # global dataset
    if args.device =='hpu':
        args.is_hpu = True
    else:
        args.is_hpu = False
      # 如果检测到 .txt 数据文件不存在  
    if (not os.path.isfile(f'./../data_{args.dataset}/{args.dataset}_train.txt')) or (not os.path.isfile(f'./../data_{args.dataset}/{args.dataset}_valid.txt') or (not os.path.isfile(f'./../data_{args.dataset}/{args.dataset}_test.txt'))):
        print("Download Dataset")
        if not os.path.exists(f'./../data_{args.dataset}'):
            os.makedirs(f'./../data_{args.dataset}')
        preprocess_raw_5core(args.dataset)# 自动调用之前那个数据预处理函数
    dataset = data_partition(args.dataset, args)# 加载处理好的数据
    
    
    [user_train, user_valid, user_test, usernum, itemnum, eval_set] = dataset
    print('user num:', usernum, 'item num:', itemnum)
    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    if args.device =='hpu':
        ###GAUDI
        import habana_frameworks.torch.core as htcore
        args.device = torch.device('hpu')
        
        # IF nn.Embedding Error solve in Gaudi, then remove this command
        args.nn_parameter = True
    elif args.device != 'hpu' and args.device != 'cpu':
        args.device = 'cuda:'+str(args.device)
    
    # dataloader
    # 1. 采样器
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)       
    # 这是一个多线程采样器，负责在训练时源源不断地生成 (User, Seq, Pos, Neg) 数据批次。
    # model init
    # 2. 模型初始化
    model = SASRec(usernum, itemnum, args).to(args.device)
    # 3. 权重初始化
    #使用 Xavier Normal 初始化参数，有助于模型收敛。
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass
    
    epoch_start_idx = 1
    #4. 断点续训与推理模式 (Resume & Inference)
    if args.state_dict_path is not None:
        # 如果指定了路径，加载预训练好的权重
        try:
            kwargs, checkpoint = torch.load(args.state_dict_path)
            kwargs['args'].device = args.device
            model = SASRec(**kwargs).to(args.device)
            model.load_state_dict(checkpoint)
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
    
    if args.inference_only:
        # 如果只是想测试效果，跳过训练环节
        # save_eval(model, dataset, args)

        print('Evaluate')
        
        
        t_test = evaluate(model, dataset, args, ranking = 10)
        print('')
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
                
        t_test = evaluate(model, dataset, args, ranking = 20)
        print('')
        print('test (NDCG@20: %.4f, HR@20: %.4f)' % (t_test[0], t_test[1]))
        
                
        sys.exit("Terminating Inference")
        
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    time_list = []
    loss_list = []
    T = 0.0
    t0 = time.time()
    start_time = time.time()
    #5. 训练循环 (Training Loop) - 核心部分
    #这是模型学习知识的地方。使用的是 BCE Loss (二元交叉熵损失)。
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):
        model.train()
        epoch_s_time = time.time()
        total_loss, count = 0, 0
        if args.inference_only: break
        # --- 一个 Epoch 内的迭代 ---
        for step in range(num_batch):
            # 1. 获取一个 Batch 的数据
            # u: 用户, seq: 历史序列
            # pos: 正样本 (真实发生的下一次购买)
            # neg: 负样本 (随机采样的未购买商品)
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            # 2. 前向传播 (Forward)
            # 得到正样本得分 (pos_logits) 和负样本得分 (neg_logits)
            pos_logits, neg_logits = model(u, seq, pos, neg)
           
            # 3. 计算损失 (Loss)
            # 目标：让 pos_logits 接近 1 (True)，让 neg_logits 接近 0 (False)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # 4. 反向传播与更新 (Backward & Step)
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            #训练逻辑：这是一个经典的自监督学习任务。
            #模型看序列 [A, B, C]。
            #希望它预测下一个是 D (正样本)，而不是随机的 X (负样本)。
            #通过不断拉大 D 的得分和 X 的得分差距，模型学会了用户的兴趣。
            #nn.Embedding
            if args.nn_parameter:
                loss += args.l2_emb * torch.norm(model.item_emb)
            else:
                for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
             
            #GAUDI
            
            loss.backward()
            if args.is_hpu:
                htcore.mark_step()
            adam_optimizer.step()
            if args.is_hpu:
                htcore.mark_step()
            
            total_loss += loss.item()
            count+=1
            
            if step % 100 == 0:
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))
        
        epoch_e_time = time.time()
        time_list.append(epoch_e_time - epoch_s_time)
        loss_list.append(total_loss/count)
    
        if epoch == args.num_epochs:# 只在最后一个 Epoch 保存模型
            folder = args.dataset
            fname = 'SASRec_saving.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            if not os.path.exists(os.path.join(folder, fname)):
                try:
                    os.makedirs(os.path.join(folder))
                except:
                    print()
            # 保存两样东西：
            # 1. model.kwargs: 模型的配置参数 (维度、层数等)，方便下次初始化
            # 2. model.state_dict(): 训练好的权重参数
            torch.save([model.kwargs, model.state_dict()], os.path.join(folder, fname))
    
    sampler.close()
    end_time = time.time()
    
    save_eval(model, dataset, args)
    
    print("Done")
    print("Time:", end_time-start_time)
#总结
#这个脚本是 LLM-SRec 项目的第一阶段。
#输入：preprocess_raw_5core 生成的 .txt 序列数据。
#过程：运行标准的 SASRec 训练（让模型学会根据序列预测下一个商品）。
#输出：一个训练好的 .pth 模型文件。

"""在代码的这一行：
torch.save([model.kwargs, model.state_dict()], os.path.join(folder, fname))
这里保存的是一个包含两部分内容的 Python 列表 (List)。
这就好比保存一个游戏存档，不仅保存了你的“等级和装备”（权重），还保存了你的“职业和角色设定”（模型配置），以便下次能完美复原。
具体包含以下两项：
1. model.kwargs (模型的“图纸”)
数据类型: 字典 (dict)
来源: 在 SASRec 类的 __init__ 函数中定义的：
Python
self.kwargs = {'user_num': user_num, 'item_num':item_num, 'args':args}
内容:
user_num: 用户总数。
item_num: 商品总数。
args: 所有的超参数配置（如 hidden_units=64, maxlen=128, num_blocks=2 等）。
作用: 用于重建模型骨架。 当你下次想加载这个模型时，你首先需要实例化 SASRec 类。如果没有这组参数，你根本不知道要创建一个多大的模型（Embedding 层开多大？Transformer 叠几层？）。有了 kwargs，就可以用 model = SASRec(**kwargs) 直接还原出一模一样的空模型结构。
2. model.state_dict() (模型的“大脑/权重”)
数据类型: 有序字典 (OrderedDict)
内容: 模型中所有可训练参数 (Parameters) 的具体数值。
item_emb.weight: 训练好的商品向量矩阵。
pos_emb.weight: 位置向量矩阵。
attention_layers...weight: Transformer 里的 Attention 和 FFN 的权重矩阵。
作用: 填充模型骨架。 这是模型经过几百个 Epoch 训练后学到的“知识”。有了骨架后，调用 model.load_state_dict(checkpoint) 就能把这些参数填进去，恢复模型的智力。
为什么不直接保存 state_dict？
#通常 PyTorch 官方教程建议直接保存 state_dict。但这个项目采用保存 [kwargs, state_dict] 的方式，是为了方便移植。
在第二阶段（训练 LLM-SRec）时，代码需要加载这个预训练好的 SASRec。 如果不保存 kwargs，你在运行第二阶段代码时，就必须手动把 SASRec 的参数（比如 hidden_units 是 64 还是 128）再从命令行敲一遍，一旦敲错（比如训练时用的 64，加载时敲了 128），模型结构就会不匹配，加载权重时就会报错。
保存 kwargs 实现了“自描述”，让 .pth 文件自己告诉代码它是怎么配置的。"""