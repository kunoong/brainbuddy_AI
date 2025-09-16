import torch.nn as nn
from torchvision import models


class CNNEncoder(nn.Module):
    def __init__(self, output_dim=512, dropout2d=0.1, proj_dropout=0.4):
        super().__init__()
        w = models.MobileNet_V3_Large_Weights.DEFAULT
        backbone = models.mobilenet_v3_large(weights=w)

        self.features = backbone.features                
        self.feat_channels = backbone.classifier[0].in_features 

        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))      
        self.drop2d  = nn.Dropout2d(dropout2d)

        flat_dim = self.feat_channels * 2 * 2            
        
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 256), nn.GELU(), nn.Dropout(proj_dropout),
            nn.Linear(256, output_dim), nn.GELU()
        )

    def forward(self, x): 
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.features(x)                 
        x = self.avgpool(x)                
        x = self.drop2d(x)
        x = x.view(B*T, -1)                 
        x = self.fc(x)                       
        return x.view(B, T, -1)              




import torch.nn as nn
from torchvision import models


class CNNEncoder(nn.Module):
    def __init__(self, output_dim=512, dropout2d=0.1, proj_dropout=0.4):
        super().__init__()
        
        # Backbone (MobileNetV3-Large)
        w = models.MobileNet_V3_Large_Weights.DEFAULT
        backbone = models.mobilenet_v3_large(weights=w)
        self.features = backbone.features
        self.feat_channels = backbone.classifier[0].in_features

        # Pooling & Dropout
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.drop2d = nn.Dropout2d(dropout2d)

        # Fully-connected projection
        flat_dim = self.feat_channels * 2 * 2
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(256, output_dim),
            nn.GELU()
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        # Backbone feature extraction
        x = x.view(B * T, C, H, W)
        x = self.features(x)

        # Pooling & dropout
        x = self.avgpool(x)
        x = self.drop2d(x)

        # Projection (flatten + FC)
        x = x.view(B * T, -1)
        x = self.fc(x)

        return x.view(B, T, -1)
