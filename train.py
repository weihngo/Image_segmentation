# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:39:55 2019

@author: Administrator
"""

# coding: utf-8

import numpy as np
import sys
import torch.nn.functional as F
from datetime import datetime

import argparse
import logging

from model import Unet,deeplabv3plus_resnet50,deeplabv3plus_resnet101,DANet,HRNet,deeplabv3plus_xception
from utils import BasicDataset,CrossEntropy,PolyLR,FocalLoss
from eval import eval_net
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

kwargs={'map_location':lambda storage, loc: storage.cuda(1)}
def load_GPUS(model,model_path,kwargs):
    state_dict = torch.load(model_path,**kwargs)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict['net'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model


correct_ratio = []
alpha = 0.5
batch_size = 32

sate_dataset_train = BasicDataset("./data/train.lst")#读取训练集文件，数据预处理在此类�?
train_steps = len(sate_dataset_train)
sate_dataset_val= BasicDataset("./data/val.lst")
train_dataloader = DataLoader(sate_dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)#将训练集封装成data_loader
eval_dataloader = DataLoader(sate_dataset_val, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)#将验证集封装�?

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
teach_model = deeplabv3plus_resnet50(num_classes=8, output_stride=16)
teach_model.to(device=device)
teach_model.load_state_dict(torch.load('./checkpoints/0.7669_best_score_model_res50_deeplabv3+.pth',map_location=device)['net'])

model = Unet(n_channels=3,n_classes=8)
model.to(device=device)
#model_dir = './checkpoints/best_score_model_unet.pth'
model_dir = './checkpoints/student_net.pth'

if os.path.exists(model_dir):
    #model = load_GPUS(model_dir, model_dir, kwargs)
    model.load_state_dict(torch.load(model_dir)['net'])
    print("loading model sccessful----" + model_dir)
#model.load_state_dict(torch.load('teach_net_params_0.9895.pkl'))
criterion = nn.CrossEntropyLoss()
criterion2 = nn.KLDivLoss()

optimizer = optim.Adam(model.parameters(),lr = 0.0001)
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
n_size = batch_size*256*256

writer = SummaryWriter(comment='student'+f'LR_0.0001_BS_32')#创建一个tensorboard文件
epochs = 50
global_step = 1
for epoch in range(epochs):
    loss_sigma = 0.0
    correct = 0.0
    total = 0.0
    model.train()
    with tqdm(total=train_steps, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        for i, data in enumerate(train_dataloader):
            inputs, labels = data['image'], data['mask']
            inputs = inputs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss1 = criterion(outputs, labels)
            
            teacher_outputs = teach_model(inputs.float())
            T = 2
            outputs_S = F.softmax(outputs/T,dim=1)
            outputs_T = F.softmax(teacher_outputs/T,dim=1)
            loss2 = criterion2(outputs_S,outputs_T)*T*T
            
            loss = loss1*(1-alpha) + loss2*alpha

    #        loss = loss1
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, dim = 1)
            correct = (predicted.cpu()==labels.cpu()).squeeze().sum().numpy()
            pbar.set_postfix(**{'loss_avg': loss.item(),'Acc': correct/n_size})
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('acc/train', correct, global_step)
            # print('loss_avg:{:.2}   Acc:{:.2%}'.format(loss_avg, correct/n_size/100))
            pbar.update(inputs.shape[0])
            global_step += 1
    if epoch % 1 == 0:
        loss_sigma = 0.0
        correct = 0
        cls_num = 10
        conf_mat = np.zeros([cls_num, cls_num])  # 混淆矩阵
        model.eval()
        for i, data in enumerate(eval_dataloader):

            # 获取图片和标�?
            inputs, labels = data['image'], data['mask']
            inputs = inputs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)
            # forward
            outputs = model(inputs)
            outputs.detach_()
            
            # 计算loss
            loss = criterion(outputs, labels)
            loss_sigma += loss.item()

            # 统计
            _, predicted = torch.max(outputs.data, 1)
            # labels = labels.data    # Variable --> tensor
            correct += (predicted.cpu()==labels.cpu()).squeeze().sum().numpy()

        avg_correct = correct/len(eval_dataloader)/n_size
        val_loss,pixel_acc_avg,mean_iou_avg,fw_iou_avg = eval_net(model,eval_dataloader,device)
        writer.add_scalar('Loss/test', loss_sigma/len(eval_dataloader), global_step)
        writer.add_scalar('fw_iou/test', fw_iou_avg, global_step)
        writer.add_scalar('acc/test',avg_correct , global_step)

        if epoch==0:
            _fw_iou_avg = fw_iou_avg
            net_save_path = 'checkpoints/student_net' + '.pth'
            model_file = {'net':model.state_dict(),'correct':correct/len(eval_dataloader),'epoch':epoch+1}
            torch.save(model_file,net_save_path)
            print('-------------------------{} set correct:{:.4%}---------------------'.format('Valid', avg_correct))
            print('-------------------------{} set fw_iou:{:.4%}---------------------'.format('Valid', fw_iou_avg))
        elif fw_iou_avg > _fw_iou_avg:
            _fw_iou_avg = fw_iou_avg
            net_save_path = 'checkpoints/student_net' + '.pth'
            model_file = {'net':model.state_dict(),'correct':correct/len(eval_dataloader),'epoch':epoch+1}
            torch.save(model_file,net_save_path)
            print('-------------------------{} set correct:{:.4%}---------------------'.format('Valid', avg_correct))
            print('-------------------------{} set fw_iou:{:.4%}---------------------'.format('Valid', fw_iou_avg))

