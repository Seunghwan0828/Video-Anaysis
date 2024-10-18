import torch
import torch.nn as nn

class CustomMViTv2Model(nn.Module):
    def __init__(self, num_classes=5, model_variant='mvit_v2_s', pretrained=True, dropout_rate=0.5):
        super(CustomMViTv2Model, self).__init__()
        
        self.model = torch.hub.load('facebookresearch/pytorchvideo', model_variant, pretrained=pretrained)
        
       
        num_features = self.model.head.proj.in_features
        
        
        self.dropout = nn.Dropout(p=dropout_rate)
        
        
        self.model.head.proj = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        return x
