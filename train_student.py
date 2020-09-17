#-*-conding:utf-8-*-
import argparse
import logging
from model import Unet,deeplabv3plus_resnet50,deeplabv3plus_resnet101,DANet,HRNet,deeplabv3plus_mobilenet
from utils import BasicDataset,CrossEntropy,PolyLR,FocalLoss
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import os
from eval import eval_net
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def train_net(unet,device,batch_size,epochs,lr,dir_checkpoint,checkpoint_name):
    global_step = 0
    writer = SummaryWriter(comment=checkpoint_name+f'LR_{lr}_BS_{batch_size}')#创建一个tensorboard文件
    sate_dataset_train = BasicDataset("./data/train.lst")#读取训练集文件，数据预处理在此类中
    sate_dataset_val= BasicDataset("./data/val.lst")
    train_steps = len(sate_dataset_train)
    train_dataloader = DataLoader(sate_dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)#将训练集封装成data_loader
    eval_dataloader = DataLoader(sate_dataset_val, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)#将验证集封装成data_loader，drop_last是将最后一个batch不足32的丢弃
    criterion = nn.CrossEntropyLoss()#交叉熵损失函数
    #criterion = CrossEntropy() #交叉熵损失函数
    #criterion = FocalLoss()#focalloss损失函数
    #optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)#优化器
    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-8,momentum = 0.9)  # 优化器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=80,factor=0.9,min_lr=5e-5)#学习率调整器
    #scheduler = PolyLR(optimizer, 8*100000/batch_size, power=0.9)
    epoch_val_loss = float('inf')#为了保存最佳模型，以验证集精度为标准
    fw_iou_avg = 0
    for epoch in range(epochs):
        epochs_loss=0#计算每个epoch的loss
        with tqdm(total=train_steps, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for idx, batch_samples in enumerate(train_dataloader):
                batch_image, batch_mask = batch_samples["image"], batch_samples["mask"]
                batch_image=batch_image.to(device=device, dtype=torch.float32)
                logits=unet(batch_image)      #torch.Size([batchsize, 8, 256, 256])
                y_true=batch_mask.to(device=device, dtype=torch.long)       #torch.Size([batchsize, 256, 256])
                loss=criterion(logits,y_true)
                epochs_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)#梯度裁剪
                optimizer.step()
                pbar.update(batch_image.shape[0])#进度条的总轮数，默认为10
                global_step += 1
                scheduler.step(loss)  # 监控量，调整学习率
                #scheduler.step()
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                if global_step % (train_steps // ( batch_size)) == 0:
                    for tag,value in net.named_parameters():
                        tag=tag.replace('.','/')
                        writer.add_histogram('weights/'+tag,value.data.cpu().numpy(),global_step)
                        writer.add_histogram('grads/' + tag, value.data.cpu().numpy(), global_step)
                    val_loss,pixel_acc_avg,mean_iou_avg,_fw_iou_avg = eval_net(net,eval_dataloader,device)
                    if fw_iou_avg<_fw_iou_avg:
                        fw_iou_avg=_fw_iou_avg
                    logging.info('Validation cross entropy: {}'.format(val_loss))
                    writer.add_scalar('Loss/test', val_loss, global_step)
                    writer.add_scalar('pixel_acc_avg', pixel_acc_avg, global_step)
                    writer.add_scalar('mean_iou_avg', mean_iou_avg, global_step)
                    writer.add_scalar('fw_iou_avg', fw_iou_avg, global_step)


        #以下将每个验证集损失保存到模型文件中，每个epoch之后取出与当前损失进行比较，当取出损失大于当前损失时，保存模型
        if os.path.exists(dir_checkpoint+checkpoint_name): #如果已经存在模型文件
            checkpoint = torch.load(dir_checkpoint+checkpoint_name)
            print(fw_iou_avg, checkpoint['fw_iou_avg'])
            if fw_iou_avg>checkpoint['fw_iou_avg']:
                print('save!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                state ={'net':net.state_dict(),'epoch_val_score':epoch_val_loss,'fw_iou_avg':fw_iou_avg,'epochth':epoch + 1}
                torch.save(state,dir_checkpoint+checkpoint_name)
                logging.info(f'checkpoint {epoch + 1} saved!')
        else:#如果不存在模型文件
            try:
                os.mkdir(dir_checkpoint)
                logging.info('create checkpoint directory!')
            except OSError:
                logging.info('save checkpoint error!')
            state ={'net':net.state_dict(),'epoch_val_score':epoch_val_loss,'fw_iou_avg':fw_iou_avg,'epochth':epoch + 1}
            torch.save(state, dir_checkpoint + checkpoint_name)
            logging.info(f'checkpoint {epoch + 1} saved!')
    writer.close()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=50,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=64,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-3,
                        help='Learning rate', dest='lr')
    return parser.parse_args()

kwargs={'map_location':lambda storage, loc: storage.cuda(1)}
def load_GPUS(model,model_path,kwargs):
    state_dict = torch.load(model_path,**kwargs)
    # create new OrderedDict that does not contain `module.`
    print("The model's loss is:"+str(state_dict['epoch_val_score']))
    print("The model's fw_iou_avg is:" + str(state_dict['fw_iou_avg']))
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict['net'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    print("loading model success!")
    return model

if __name__=="__main__":
    dir_checkpoint='checkpoints/'
    checkpoint_name='best_score_model_unet.pth'
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net = Unet(n_channels=3, n_classes=8, bilinear=True)
    #net = deeplabv3plus_mobilenet(num_classes=8, output_stride=16)
    #net = DANet(8, backbone='resnet50', pretrained_base=True)
    #net = HRNet(48, 8, 0.1)
    net.to(device=device)
    # state_dict = torch.load(dir_checkpoint+checkpoint_name)
    # print(state_dict['epoch_val_score'], state_dict['fw_iou_avg'])
    # net.load_state_dict(state_dict['net'])
    #net = load_GPUS(net, dir_checkpoint +checkpoint_name, kwargs)
    #net = torch.nn.DataParallel(net)
    train_net(net,device, args.batchsize, args.epochs,args.lr,dir_checkpoint,checkpoint_name)

    # import torch
    # model=deeplabv3_resnet50(num_classes=8, output_stride=16)
    # # from torchsummary import summary
    # model=model.cuda()
    # x=torch.rand((2, 3, 256, 256)).cuda()
    # # summary(model.cuda(),x)
    # y=model(x)
    # print(y.shape)



