import torch
import torch.nn as nn
from torchsummary import summary as model_summary

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.n_classes = 21

    def Conv(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
    
    def maxpool(self):
        return nn.MaxPool2d(kernel_size=2, stride=2)
    
    def block1(self, in_channels, out_channels):
        return nn.Sequential(
            self.Conv(in_channels, out_channels),
            self.Conv(out_channels, out_channels),
            self.maxpool()
        )
    
    def block2(self, in_channels, out_channels):
        return nn.Sequential(
            self.Conv(in_channels, out_channels),
            self.Conv(out_channels, out_channels),
            self.Conv(out_channels, out_channels, kernel_size=1, padding=0),
            self.maxpool()
        )
    
    def encoder(self, x):
        block1 = self.block1(3,64)(x) # 224x224x3 -> 112x112x64
        block2 = self.block1(64,128)(block1) # 112x112x64 -> 56x56x128
        block3 = self.block2(128,256)(block2) # 56x56x128 -> 28x28x256
        block4 = self.block2(256,512)(block3) # 28x28x256 -> 14x14x512
        block5 = self.block2(512,512)(block4) # 14x14x512 -> 7x7x512
        block6 = nn.Sequential(
            self.Conv(512,4096), # 7x7x512 -> 7x7x4096
            self.Conv(4096,4096) # 7x7x4096 -> 7x7x4096
        )(block5)
        output = self.Conv(4096, self.n_classes, kernel_size=3, padding=1)(block6) # 7x7x4096 -> 7x7x21
        blocks =(block3, block4) # for skip connections
        return blocks, output
    
    def decoder(self, convs, output):
        b3,b4 = convs

        decoder1 = nn.ConvTranspose2d(self.n_classes, self.n_classes, kernel_size=4, stride=2, padding=1)(output) # 7x7x21 -> 14x14x21
        skip_layer1 = nn.Conv2d(512, self.n_classes,kernel_size=3,padding=1)(b4) + decoder1 # 14x14x21 + 14x14x21 -> 14x14x21

        decoder2 = nn.ConvTranspose2d(self.n_classes, self.n_classes, kernel_size=4, stride=2, padding=1)(skip_layer1) # 28x28x21
        skip_layer2 = nn.Conv2d(256, self.n_classes,kernel_size=3,padding=1)(b3) + decoder2 # 28x28x21 + 28x28x21 -> 28x28x21
        
        output = nn.ConvTranspose2d(self.n_classes, self.n_classes, kernel_size=16,  padding=4)(skip_layer2) # 28x28x21 -> 224x224x21
        return output

    def forward(self,x):
        convs, output = self.encoder(x)
        output = self.decoder(convs,output)
        return output
    
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