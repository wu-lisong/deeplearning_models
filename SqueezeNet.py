import torch.nn as nn
import torch
import torch.nn.init as init

class Fire(nn.Module):
    def __init__(self,in_planes,squeeze_planes,expand1x1_planes,expand3x3_planes):
        super(Fire,self).__init__()
        self.in_planes = in_planes
        self.squeeze = nn.Conv2d(in_planes,squeeze_planes,kernel_size = 1)
        self.squeeze_activation = nn.ReLU(inplace = True)
        self.expand1x1 = nn.Conv2d(squeeze_planes,expand1x1_planes,kernel_size = 1)
        self.expand1x1_activation = nn.ReLU(inplace = True)
        self.expand3x3 = nn.Conv2d(squeeze_planes,expand3x3_planes,kernel_size = 3,stride = 1,padding = 1)
        self.expand3x3_activation = nn.ReLU(inplace = True)

    def forward(self,x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ],dim = 1)

class SqueezeNet(nn.Module):
    def __init__(self,num_classes = 1000):
        super(SqueezeNet,self).__init__()
        self.num_classes = num_classes

        self.features = torch.nn.Sequential(
            nn.Conv2d(3,96,kernel_size = 7,stride = 2,padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3,stride = 2,ceil_mode = True),
            Fire(96,16,64,64),
            Fire(128,16,64,64),
            Fire(128,32,128,128),
            nn.MaxPool2d(kernel_size = 3,stride = 2,ceil_mode = True),
            Fire(256,32,128,128),
            Fire(256,48,192,192),
            Fire(384,48,192,192),
            Fire(384,64,256,256),
            nn.MaxPool2d(kernel_size = 3,stride = 2,ceil_mode = True),
            Fire(512,64,256,256)
        )

        final_conv = nn.Conv2d(512,self.num_classes,kernel_size = 1)
        self.classifier = torch.nn.Sequential(
            nn.Dropout(p = 0.5),
            final_conv,
            nn.ReLU(inplace = True),
            nn.AdaptiveAvgPool2d(output_size = (1,1))
        )

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight,mean = 0.0,std = 0.01)
                else:
                    init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias,0)
    
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x,1)




if __name__ == "__main__":
    model = SqueezeNet()
    tensor = torch.randn(1,3,224,224)
    result = model(tensor)
    print(result.shape)
