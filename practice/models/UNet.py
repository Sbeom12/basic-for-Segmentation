import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary as model_summary
from torchvision import transforms

def Conv(in_channels, out_channels, stride = 1, kernel_size = 3, padding = 0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size = kernel_size, padding=padding),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size = kernel_size, padding=padding),
        nn.ReLU(inplace=True)
    )
    
class UNet(nn.Module):
    def __init__(self,n_classes = 2):
        super().__init__()
        self.block1 = Conv(1,64)
        self.block2 = Conv(64,128)
        self.block3 = Conv(128,256)
        self.block4 = Conv(256,512)
        self.block5 = Conv(512,1024)
        
        self.block6 = Conv(1024,512)
        self.block7 = Conv(512,256)
        self.block8 = Conv(256,128)
        self.block9 = Conv(128,64)
        
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        
        self.final_conv = nn.Conv2d(64,n_classes, kernel_size=1, stride = 1, padding=0)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Down 1
        down1 = self.block1(x) # skip을 위해 저장
        x = self.maxpool(down1)
        x = self.relu(x)
        # Down 2
        down2 = self.block2(x) # skip을 위해 저장
        x = self.maxpool(down2)
        x = self.relu(x)
        # Down 3
        down3 = self.block3(x) # skip을 위해 저장
        x = self.maxpool(down3)
        x = self.relu(x)
        # Down 4
        down4 = self.block4(x) # skip을 위해 저장
        x = self.maxpool(down4)
        x = self.relu(x)
        # Latent Space
        x = self.block5(x)
        # Up 1 and Skip conncetion
        x = self.deconv1(x)
        cat1 = torch.cat((transforms.CenterCrop((x.shape[2], x.shape[3]))(down4), x), dim=1)   
        x = self.relu(cat1)
        x = self.block6(x)

        # Up 2 and Skip conncetion
        x = self.deconv2(x)
        cat2 = torch.cat((transforms.CenterCrop((x.shape[2], x.shape[3]))(down3), x), dim=1)
        x = self.relu(cat2)
        x = self.block7(x)

        # Up 3 and Skip conncetion
        x = self.deconv3(x)
        cat3 = torch.cat((transforms.CenterCrop((x.shape[2], x.shape[3]))(down2), x), dim=1)
        x = self.relu(cat3)
        x = self.block8(x)
        
        # Up 4 and Skip conncetion
        x = self.deconv4(x)
        cat4 = torch.cat((transforms.CenterCrop((x.shape[2], x.shape[3]))(down1), x), dim=1)
        x = self.relu(cat4)
        x = self.block9(x)

        # Segmentation Map
        x = self.final_conv(x)
        return x

if __name__ == '__main__':
    model = UNet()
    model_summary(model, input_size=(1, 224, 224), device='cpu')