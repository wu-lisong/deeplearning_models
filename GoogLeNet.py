import warnings
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ["GoogLeNet","googlenet"]

model_urls = {
    #GoogLeNet ported from Tensorflow
    "googlenet": "https://download.pytorch.org/models/googlenet-1378be20.pth"
}

_GoogLeNetOutputs = namedtuple("GoogLeNetOutputs",["logits","aux_logits2","aux_logits1"])

def googlenet(pretrained = False,progress = True,**kwargs):
    r"""GoogLeNet (Inception V1) model architecture from
    going deeper with convolutional
    Args:
        pretrained (bool): If True,returns a model pretrained on ImageNet
        progress   (bool): If True, displays a progress bar of the download tp stderr
        
    """
    if pretrained:
        if "transform_input" not in kwargs:
            kwargs["transform_input"] = True
        if "aux_logits" not in kwargs:
            kwargs["aux_logits"] = False
        if kwargs["aux_logits"]:
            warnings.warn("auxiliary heads in the pretrained googlenet model are Not pretrained, So make sure to train them.")

        original_aux_logits = kwargs["aux_logits"]
        kwargs["aux_logits"] = True
        kwargs["init_weights"] = False
        model = GoogLeNet(**kwargs)

        state_dict = load_state_dict_from_url(model_urls["googlenet"],progress = progress)
        model.load_state_dict(state_dict)

        if not original_aux_logits:
            model.aux_logits = False
            del model.aux1,model.aux2

        return model
    return GoogLeNet(**kwargs)



class BasicConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super(BasicConv2d,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,bias = False,**kwargs)
        self.bn = nn.BatchNorm2d(out_channels,eps = 0.001)

    def forward(self,x):
        x = self.bn(self.conv(x))
        return F.relu(x,inplace = True)


class InceptionAux(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(InceptionAux,self).__init__()
        self.conv = BasicConv2d(in_channels,128,kernel_size = 1)
        self.fc1 = nn.Linear(2048,1024)
        self.fc2 = nn.Linear(1024,num_classes)

    def forward(self,x):
        #aux1 : N X 512 X 14 X 14       aux2 : N X 528 X 14 X 14
        x = F.adaptive_avg_pool2d(x,(4,4))
        #aux1 : N X 512 X 4  X 4       aux2 : N X 528 X 4 X 4
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x),inplace = True)
        x = F.dropout(x,0.7,training = self.training)
        return self.fc2(x)


class Inception(nn.Module):
    def __init__(self,in_channels,ch1X1,ch3X3red,ch3X3,ch5X5red,ch5X5,pool_proj):
        super(Inception,self).__init__()
        self.branch1 = BasicConv2d(in_channels,ch1X1,kernel_size = 1,stride = 1,padding = 0)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels,ch3X3red,kernel_size = 1,stride = 1,padding = 0),
            BasicConv2d(ch3X3red,ch3X3,kernel_size = 3,stride = 1,padding = 1))
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels,ch5X5red,kernel_size = 1,stride = 1,padding = 0),
            BasicConv2d(ch5X5red,ch5X5,kernel_size = 5,stride = 1,padding = 2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3,stride = 1,padding = 1,ceil_mode = True),
            BasicConv2d(in_channels,pool_proj,kernel_size = 1,stride = 1,padding = 0)
        )

    def forward(self,x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        return torch.cat((branch1,branch2,branch3,branch4),dim = 1)

class GoogLeNet(nn.Module):
    def __init__(self,num_classes = 1000,aux_logits = True,transform_input = False,init_weights = True):
        super(GoogLeNet,self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.init_weights = init_weights

        self.conv1 = BasicConv2d(3,64,kernel_size = 7,stride = 2,padding = 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 3,stride = 2,ceil_mode = True)
        self.conv2 = BasicConv2d(64,64,kernel_size = 1)
        self.conv3 = BasicConv2d(64,192,kernel_size = 3,stride = 1,padding = 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 3,stride = 2,ceil_mode = True)

        self.inception3a = Inception(192,64,96,128,16,32,32)
        self.inception3b = Inception(256,128,128,192,32,96,64)
        self.maxpool3 = nn.MaxPool2d(kernel_size = 3,stride = 2,ceil_mode = True)

        self.inception4a = Inception(480,192,96,208,16,48,64)
        self.inception4b = Inception(512,160,112,224,24,64,64)
        self.inception4c = Inception(512,128,128,256,24,64,64)
        self.inception4d = Inception(512,112,144,288,32,64,64)
        self.inception4e = Inception(528,256,160,320,32,128,128)


        self.maxpool4 = nn.MaxPool2d(kernel_size = 3,stride = 2,ceil_mode = True)
        
        self.inception5a = Inception(832,256,160,320,32,128,128)
        self.inception5b = Inception(832,384,192,384,48,128,128)

        if aux_logits:
            self.aux1 = InceptionAux(512,num_classes)
            self.aux2 = InceptionAux(528,num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024,num_classes)

        if self.init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2,2,scale = 0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()),dtype = m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.data.copy_(values)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)


    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)
        x = self.inception4e(x)

        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = x.view(x.size()[0],-1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.training and self.aux_logits:
            return _GoogLeNetOutputs(x,aux2,aux1)
        return x

if __name__ == "__main__":
    model = GoogLeNet(num_classes = 10,aux_logits = True)
    tensor = torch.randn(1,3,224,224)
    output = model(tensor)
    print(output)