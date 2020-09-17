import torch
import os
from torch.utils.data import Dataset, DataLoader
import logging
import numpy as np
from PIL import Image
class BasicDataset(Dataset):
    def __init__(self,list_path):
        self.list_path=list_path
        self.img_list = []
        self.matches = [100, 200, 300, 400, 500, 600, 700, 800]
        self.img_list=[line.strip().split(" ") for line in open(list_path)]
        print(f'Creating dataset with {len(self.img_list)} examples')

    def __len__(self):
        return len(self.img_list)

    @classmethod
    def preprocess(cls,img,mask,matches,nClasses=8):
        #process img
        img = np.float32(img) / 127.5 - 1
        mask=np.array(mask)
        #process mask
        #seg_labels = np.zeros((256, 256, nClasses))
        for m in matches:
            mask[mask == m] = matches.index(m)

       #one-hot
        # for c in range(nClasses):
        #     seg_labels[:, :, c] = (mask == c).astype(int)
        # seg_labels = np.reshape(seg_labels, (256 * 256, nClasses))
        return img,mask

    def __getitem__(self, i):
        img_file = self.img_list[i][0]
        mask_file=self.img_list[i][1]
        img = Image.open(img_file)
        mask = Image.open(mask_file)

        img,mask = self.preprocess(img,mask,self.matches)
        return {
            'image': torch.from_numpy(img).permute(2,0,1).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }
#
# if __name__ == "__main__":
#     sentiment_dataset = BasicDataset("../data/train.lst")
#     sentiment_dataloader = DataLoader(sentiment_dataset, batch_size=1,shuffle=True,num_workers=5)
#     for idx, batch_samples in enumerate(sentiment_dataloader):
#         text_batchs, text_labels = batch_samples["image"], batch_samples["mask"]
#         # print(text_batchs)
#         print(text_labels[0].shape)
#         break


       #one-hot
        # for c in range(nClasses):
        #     seg_labels[:, :, c] = (mask == c).astype(int)
        # seg_labels = np.reshape(seg_labels, (256 * 256, nClasses))
        return img,mask

    def __getitem__(self, i):
        img_file = self.img_list[i][0]
        mask_file=self.img_list[i][1]
        img = Image.open(img_file)
        mask = Image.open(mask_file)

        img,mask = self.preprocess(img,mask,self.matches)
        return {
            'image': torch.from_numpy(img).permute(2,0,1).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }
