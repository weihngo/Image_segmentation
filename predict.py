from utils import  plot_img_and_mask
from utils import  BasicDataset
from model import Unet
import matplotlib.pyplot as plt
from model import Unet,deeplabv3plus_resnet50,deeplabv3plus_resnet101
import logging
import torch
import numpy as np
import torch.nn.functional as F
import os
import cv2
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from utils import BasicDataset
from tqdm import tqdm
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def predict_img(net,ori_img,device):
    net.eval()
    img = np.float32(ori_img) / 127.5 - 1
    img=torch.from_numpy(img).permute(2, 0, 1).type(torch.FloatTensor).unsqueeze(0)
    img=img.to(device=device,dtype=torch.float32)

    with torch.no_grad():
        output=net(img)
        probs=F.softmax(output,dim=1).squeeze(0) #[8, 256, 256]
        pre=torch.argmax(probs,dim=0).cpu().data.numpy()#[256,256]
        pre_mask_img=Image.fromarray(np.uint8(pre))
        palette = [
            0, 0, 0,
            0, 0, 255,
            15, 29, 15,
            26, 141, 52,
            41, 41, 41,
            65, 105, 225,
            85, 11, 18,
            128, 0, 128,
        ]
        pre_mask_img.putpalette(palette)

        # tf = transforms.Compose(
        #     [
        #         transforms.ToPILImage(),
        #         transforms.ToTensor()
        #     ]
        # )
        # probs = tf(probs.cpu())
        # full_mask = probs.squeeze().cpu().numpy()#(3, 256, 256)

        return pre,pre_mask_img

def mask_to_image(mask):
    return (mask.transpose(1,2,0) * 255).astype(np.uint8)

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
if __name__ == "__main__":

    matches = [100, 200, 300, 400, 500, 600, 700, 800]
    dir_checkpoint = 'checkpoints/'
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    #net = Unet(n_channels=3, n_classes=8, bilinear=True)
    net = Unet(n_channels=3,n_classes=8)
    net.to(device=device)
    #net=load_GPUS(net, dir_checkpoint + 'best_score_model_res50_deeplabv3+.pth', kwargs)
    checkpoint = torch.load(dir_checkpoint + 'student_net.pth',map_location=device)
    net.load_state_dict(checkpoint['net'])
    logging.info("Model loaded !")

    list_path = "data/test.lst"
    output_path="data/results/"
    img_list = [line.strip('\n') for line in open(list_path)]
    for i, fn in tqdm(enumerate(img_list)):
        save_img = np.zeros((256, 256), dtype=np.uint16)
        logging.info("\nPredicting image {} ...".format(i))
        img = Image.open(fn)
        pre,_=predict_img(net,img,device)
        for i in range(256):
            for j in range(256):
                save_img[i][j] = matches[int(pre[i][j])]
        index=fn.split("/")[-1].split(".")[0]
        cv2.imwrite(os.path.join(output_path, index+".png"), save_img)

    # image_path='../baseline/test/images/1_1_1.tif'
    # img=Image.open(image_path)
    # mask,pre_mask_img=predict_img(net,img,device)
    # # result = mask_to_image(mask)
    # print(mask)
    #
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(pre_mask_img)
    # plt.subplot(1,2,2)
    # plt.imshow(img)
    # plt.show()




