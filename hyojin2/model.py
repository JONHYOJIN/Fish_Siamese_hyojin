import torch
import torch.nn as nn
from torchinfo import summary

class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()

        # Conv Layers
        self.conv = nn.Sequential(
            # Conv : 64 channel / 10x10 kernel | 1@105x105 -> 64@96x96
            nn.Conv2d(1, 64, 10),  
            nn.ReLU(inplace=True),
            # MaxPool : stride=2 | 64@96x96 -> 64@48x48
            nn.MaxPool2d(2),  

            # Conv : 128 channel / 7x7 kernel | 64@48x48 -> 128@42x42
            nn.Conv2d(64, 128, 7),
            nn.ReLU(inplace=True), 
            # MaxPool : stride=2 | 128@42x42 -> 128@21x21
            nn.MaxPool2d(2), 

            # Conv : 128 channel / 4x4 kernel | 128@21x21 -> 128@18x18
            nn.Conv2d(128, 128, 4),
            nn.ReLU(inplace=True), 
            # MaxPool : stride=2 | 128@18x18 -> 128@9x9
            nn.MaxPool2d(2),  

            # Conv : 256 channel / 4x4 kernel | 128@9x9 -> 256@6x6 
            nn.Conv2d(128, 256, 4),
            nn.ReLU(inplace=True),  

            # (참고: 256x6x6 = 9216)
        )

        # Fully Connected Layer
        self.linear = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
        
        # Output Layer (Fully Connected Layer)
        self.out = nn.Sequential(nn.Linear(4096, 1), nn.Sigmoid())

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_uniform_(m.weight)
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.normal_(m.bias, mean=0.5, std=0.01)
            elif isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, mean=0, std=0.2)
                nn.init.normal_(m.bias, mean=0.5, std=0.01)

    def sub_forward(self, x):
        # Conv Layers
        x = self.conv(x)
        # flatten to 9216
        x = x.view(x.size()[0], -1)
        # Fully Connected Layers : 9216 -> 4096
        x = self.linear(x)
        return x

    def forward(self, x1, x2):
        """
            1. 각 2개의 이미지의 4096 차원의 결과를 도출(h1, h2)
            2. 두 결과의 L1 distance 도출(diff)
            3. diff를 기반으로 유사도를 계산; 4096 -> 1 (scores)
        """
        # 1.
        h1 = self.sub_forward(x1)
        h2 = self.sub_forward(x2)

        # 2.
        diff = torch.abs(h1 - h2)

        # 3.
        scores = self.out(diff)

        return scores

if __name__=='__main__':
    model = SiameseNet()
    print(summary(model))