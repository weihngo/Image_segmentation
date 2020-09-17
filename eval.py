import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils import Evaluator
from model import Unet
from utils import BasicDataset
from torch.utils.data import Dataset, DataLoader
def eval_net(net,data_loader,device):
    net.eval()
    val_batch_num=len(data_loader)
    eval_loss=0

    e = Evaluator(num_class=8)
    pixel_acc_avg = 0
    mean_iou_avg = 0
    fw_iou_avg = 0

    with tqdm(total=val_batch_num, desc='Validation round', unit='batch', leave=False) as pbar:
        for idx,batch_samples in enumerate(data_loader):
            batch_image, batch_mask = batch_samples["image"], batch_samples["mask"]
            batch_image=batch_image.to(device=device,dtype=torch.float32)
            mask_true=batch_mask.to(device=device,dtype=torch.long)

            with torch.no_grad():
                mask_pred=net(batch_image)
                probs = F.softmax(mask_pred, dim=1).squeeze(0)  # [8, 256, 256]
                pre = torch.argmax(probs, dim=1)  # [256,256]

            #????
            e.add_batch(mask_true.cpu().data.numpy(),pre.cpu().data.numpy())
            pixel_acc=e.Pixel_Accuracy()
            pixel_acc_avg+=pixel_acc

            mean_iou=e.Mean_Intersection_over_Union()
            mean_iou_avg+=mean_iou

            fw_iou=e.Frequency_Weighted_Intersection_over_Union()
            fw_iou_avg+=fw_iou

            eval_loss+=F.cross_entropy(mask_pred,mask_true).item()
            pbar.set_postfix({'eval_loss': eval_loss/(idx+1)})
            pbar.update()
            e.reset()
    print("pixel_acc_avg:"+str(pixel_acc_avg/val_batch_num))
    print("mean_iou_avg:" + str(mean_iou_avg / val_batch_num))
    print("fw_iou_avg:" + str(fw_iou_avg / val_batch_num))
    net.train()
    return eval_loss/val_batch_num,pixel_acc_avg/val_batch_num,mean_iou_avg / val_batch_num,fw_iou_avg / val_batch_num

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

if __name__=="__main__":
    dir_checkpoint = 'checkpoints/'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = Unet(n_channels=3, n_classes=8, bilinear=True)
    net.to(device=device)
    model = torch.load(dir_checkpoint + 'best_score_model_unet.pth')
    net.load_state_dict(model['net'])
    #net = load_GPUS(net, dir_checkpoint + 'student_net.pth', kwargs)
    sate_dataset_val = BasicDataset("./data/val.lst")
    eval_dataloader = DataLoader(sate_dataset_val, batch_size=32, shuffle=True, num_workers=5, drop_last=True)
    print("begin")
    eval_net(net, eval_dataloader, device)