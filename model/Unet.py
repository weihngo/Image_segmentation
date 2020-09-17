import torch.nn.functional as F
from .Unet_module import *
class Unet(nn.Module):
    def __init__(self,n_channels,n_classes,bilinear=True):
        super(Unet,self).__init__()
        self.n_channels=n_channels
        self.n_classes=n_classes
        self.bilinear=bilinear

        self.conv_block=Conv_block(n_channels,64)
        self.down1=Down_block(64,128)
        self.down2=Down_block(128,256)
        self.down3=Down_block(256,512)

        factor=2 if bilinear else 1
        self.down4=Down_block(512,1024//factor)

        self.up1=Up_block(1024,512//factor,bilinear)
        self.up2 = Up_block(512, 256// factor, bilinear)
        self.up3 = Up_block(256, 128// factor, bilinear)
        self.up4 = Up_block(128, 64, bilinear)
        self.output=OutConv(64,n_classes)

    def forward(self,x):
        x1=self.conv_block(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.output(x)
        return logits