import torch
import torch.utils.model_zoo as model_zoo
 
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]
 
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(torch.nn.Module):
    def __init__(self,features,num_classes = 1000,init_weights = False):
        super(VGG,self).__init__()
        self.num_classes = num_classes
        self.init_weights = init_weights
        self.features = features
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512*7*7,4096),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096,4096),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096,self.num_classes)
        )
        if self.init_weights:
            self._initialize_weights()

    def forward(self,x):
        out = self.features(x)
        out = out.view(out.size()[0],-1)
        out = self.classifier(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight,mode = "fan_out",nonlinearity = 'relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias,0)
            elif isinstance(m,torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight,1)
                torch.nn.init.constant_(m.bias,0)
            elif isinstance(m,torch.nn.Linear):
                torch.nn.init.normal_(m.weight,0.01)
                torch.nn.init.constant_(m.bias,0)

def make_layers(cfg,batch_norm = False):
    layers = []
    in_channels = 3

    for v in cfg:
        if v == "M":
            layers += [torch.nn.MaxPool2d(kernel_size = 2,stride = 2)]
        else:
            conv2d = torch.nn.Conv2d(in_channels = in_channels,out_channels = v,kernel_size = 3,stride = 1,padding = 1)
            if batch_norm:
                layers += [conv2d,torch.nn.BatchNorm2d(v),torch.nn.ReLU(inplace = True)]
            else:
                layers += [conv2d,torch.nn.ReLU(inplace = True)]
            in_channels = v

    return torch.nn.Sequential(*layers)

cfg = {
    'A':[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'B':[64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'D':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
    'E':[64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M']
}

def vgg11(pretrained = False,**kwargs):
    """VGG 11-layer model(configuration "A")
    Args:
        pretrained (bool): if True,returns a model pretrained on ImageNet
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfg = cfg['A']),**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model

def vgg11_bn(pretrained = False,**kwargs):
    """VGG 11-layer model(configuration "A") with batch normalization
    Args:
        pretrained (bool): if True,returns a model pretrained on ImageNet
    """  
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']),**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model

def vgg13(pretrained = False,**kwargs):
    """VGG 13-layer model(configuration "B")
    Args:
        pretrained (bool): if True,returns a model pretrained on ImageNet
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfg = cfg['B']),**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model

def vgg13_bn(pretrained = False,**kwargs):
    """VGG 13-layer model(configuration "B") with batch normalization
    Args:
        pretrained (bool): if True,returns a model pretrained on ImageNet
    """  
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']),**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model

def vgg16(pretrained = False,**kwargs):
    """VGG 16-layer model(configuration "D")
    Args:
        pretrained (bool): if True,returns a model pretrained on ImageNet
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfg = cfg['D']),**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model

def vgg16_bn(pretrained = False,**kwargs):
    """VGG 16-layer model(configuration "D") with batch normalization
    Args:
        pretrained (bool): if True,returns a model pretrained on ImageNet
    """  
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']),**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model

def vgg19(pretrained = False,**kwargs):
    """VGG 19-layer model(configuration "E")
    Args:
        pretrained (bool): if True,returns a model pretrained on ImageNet
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfg = cfg['E']),**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model

def vgg19_bn(pretrained = False,**kwargs):
    """VGG 19-layer model(configuration "E") with batch normalization
    Args:
        pretrained (bool): if True,returns a model pretrained on ImageNet
    """  
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']),**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = vgg16(pretrained = False).to(device)
    image = torch.randn(1,3,224,224).to(device)
    with torch.no_grad():
        logit = model(image)
    print(logit.shape)
