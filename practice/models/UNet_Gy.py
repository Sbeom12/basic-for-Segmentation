import torch
import torch.nn as nn
from torchsummary import summary as model_summary

def Conv(in_channels, out_channels, stride = 1, kernel_size = 3, padding = 0):
    return nn.Sequntial(
        nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size = kernel_size, padding=padding),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size = kernel_size, padding=padding),
        nn.ReLU(inplace=True),
    )
    
class UNet(nn.Module):
    def __init__(self):
        self.block1 = Conv(1,64)
        self.block2 = Conv(64,128)
        self.block3 = Conv(128,256)
        self.block4 = Conv(256,512)
        self.block5 = Conv(512,1024)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        
        self.conv1 = nn.Conv2d(64,2,kernel_size=1, stride = 1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Down 1
        x1 = self.block1(x) # skip을 위해 저장
        x = self.maxpool(x1)
        # Down 2
        x2 = self.block2(x) # skip을 위해 저장
        x = self.maxpool(x2)
        # Down 3
        x3 = self.block3(x) # skip을 위해 저장
        x = self.maxpool(x3)
        # Down 4
        x4 = self.block4(x) # skip을 위해 저장
        x = self.maxpool(x4)
        
        # Up 1
        x = self.block5(x)
        x = self.deconv1(x)
        x = self.relu(x)
        # Up-Skip? How to Crop?