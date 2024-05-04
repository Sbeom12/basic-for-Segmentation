import torch
import torch.nn as nn
from torchsummary import summary as model_summary

def Conv(in_channels, out_channels, kernel_size=3, padding=1, stride=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    ]

def block1(in_channels, out_channels):
    conv1 = Conv(in_channels, out_channels)
    conv2 = Conv(out_channels, out_channels)
    combined = conv1 + conv2
    return nn.Sequential(
        *combined,
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    
def block2(in_channels, out_channels):
    conv1 = Conv(in_channels, out_channels)
    conv2 = Conv(out_channels, out_channels)
    conv3 = Conv(out_channels, out_channels, kernel_size=1, padding=0)
    combined = conv1 + conv2 + conv3
    return nn.Sequential(
        *combined,
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
def block3(in_channels, out_channels):
    conv1 = Conv(in_channels,out_channels)
    conv2 = Conv(out_channels,out_channels)
    combined = conv1 + conv2
    return nn.Sequential
class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = block1(3,64)
        self.block2 = block1(64,128)
        self.block3 = block1(128,256)
        self.block4 = block1(256,512)
        self.block5 = block1(512,512)
        conv1 = Conv(512,4096)
        conv2 = Conv(4096,4096)
        self.conv1 = nn.Sequential(*conv1,*conv2)
        self.conv2 = nn.Sequential(*Conv(4096,self.n_classes))
        
        # block6 = nn.Sequential(
        #     self.Conv(512,4096), # 7x7x512 -> 7x7x4096
        #     self.Conv(4096,4096) # 7x7x4096 -> 7x7x4096
        # )
        # output = self.Conv(4096, self.n_classes, kernel_size=3, padding=1)(block6) # 7x7x4096 -> 7x7x21
        
        
        self.n_classes = 21
        self.deconv1 = nn.ConvTranspose2d(self.n_classes, self.n_classes, kernel_size=4, stride=2, padding=1) # 7x7x21 -> 14x14x21
        self.deconv2 = nn.ConvTranspose2d(self.n_classes, self.n_classes, kernel_size=4, stride=2, padding=1) # 28x28x21
        self.deconv3 = nn.ConvTranspose2d(self.n_classes, self.n_classes, kernel_size=16,  padding=4) # 28x28x21 -> 224x224x21
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(self.n_classes)
    def forward(self, x):
        # encoder
        x = self.block1(x)
        x1 = x
        x = self.block2(x)
        x2 = x
        x = self.block3(x)
        x3 = x
        x = self.block4(x)
        x4 = x
        x = self.block5(x)
        x5 = x
        x = self.conv1(x)
        x = self.conv2(x)
        
        
        # decoder
        x = self.deconv1(x) # 7x7x21 -> 14x14x21
        x = nn.Conv2d(512, self.n_classes,kernel_size=3,padding=1)(x4) + x
        x = self.deconv2(x)
        x = nn.Conv2d(256, self.n_classes,kernel_size=3,padding=1)(x3) + x
        x = self.deconv3(x)
        
        return x
    
if __name__=="__main__":
    # Create an instance of the FCN model
    model = FCN()
    model_summary(model, (3,224,224), device='cpu')

    # Define your loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Load your data and train the model
    # ..
    # torch.save(model.state_dict(), 'fcn_model.pth')