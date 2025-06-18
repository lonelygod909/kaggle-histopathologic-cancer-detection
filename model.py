import torch
import torch.nn as nn
from predict_head import PredictHead
import torchvision
import timm
import hiera

# 可以选择多种基准模型
class Classifier(nn.Module):
    def __init__(self, num_classes, feature_dim=1000, hidden_dim=512, model_name='resnet18', pretrained=True):        
        super().__init__()

        if model_name == 'resnet18':
            self.encoder = torchvision.models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            self.encoder = torchvision.models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
        elif model_name == 'resnet152':
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            
        elif model_name == 'resnext50_32x4d':
            self.encoder = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        elif model_name == 'resnext101_32x8d':
            self.encoder = torchvision.models.resnext101_32x8d(pretrained=pretrained)
            
            
        elif model_name == 'densenet121':
            self.encoder = torchvision.models.densenet121(pretrained=pretrained)
        elif model_name == 'densenet169':
            self.encoder = torchvision.models.densenet169(pretrained=pretrained)
        elif model_name == 'densenet201':
            self.encoder = torchvision.models.densenet201(pretrained=pretrained)

            
        elif model_name == 'convnext_tiny':
            self.encoder = torchvision.models.convnext_tiny(pretrained=pretrained)
        elif model_name == 'convnext_small':
            self.encoder = torchvision.models.convnext_small(pretrained=pretrained)
        elif model_name == 'convnext_base':
            self.encoder = torchvision.models.convnext_base(pretrained=pretrained)
            
            
        elif model_name == 'seresnext50_32x4d':
            self.encoder = timm.create_model('seresnext50_32x4d', pretrained=pretrained)
            feature_dim = 2048
        elif model_name == 'seresnext101_32x4d':
            self.encoder = timm.create_model('seresnext101_32x4d', pretrained=pretrained)
        elif model_name == 'tf_efficientnet_b0':
            self.encoder = timm.create_model('tf_efficientnet_b0', pretrained=pretrained)
        elif model_name == 'tf_efficientnet_b4':
            self.encoder = timm.create_model('tf_efficientnet_b4', pretrained=pretrained)
        elif model_name == 'tf_efficientnetv2_s':
            self.encoder = timm.create_model('tf_efficientnetv2_s', pretrained=pretrained)
        elif model_name == 'swin_tiny_patch4_window7_224':
            self.encoder = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained)
        elif model_name == 'swin_small_patch4_window7_224':
            self.encoder = timm.create_model('swin_small_patch4_window7_224', pretrained=pretrained)
        
        elif model_name == 'hiera_tiny':
            self.encoder = hiera.Hiera.from_pretrained("facebook/hiera_tiny_224.mae_in1k_ft_in1k")  
            
        elif model_name == 'dino_vits8':
            self.encoder = torch.hub.load('facebookresearch/dino:main', 'dino_vits8', pretrained=pretrained)
            feature_dim = 384
        elif model_name == 'dino_vits16':
            self.encoder = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=pretrained)
            feature_dim = 384
        elif model_name == 'dino_vitb8':
            self.encoder = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8', pretrained=pretrained)
            feature_dim = 768
        elif model_name == 'dino_vitb16':
            self.encoder = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=pretrained)
            feature_dim = 768
            
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        self.predict_head = PredictHead(feature_dim, num_classes, hidden_dim=hidden_dim)
        

    def forward(self, x):
    
        features = self.encoder(x)
        
        outputs = self.predict_head(features) 

        return outputs  
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
            
    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def create_model(num_classes, model_name='resnet34', freeze_encoder=True, pretrained=True):

    model = Classifier(num_classes=num_classes, model_name=model_name, pretrained=pretrained)
    
    if freeze_encoder:
        model.freeze_encoder()
    
    return model