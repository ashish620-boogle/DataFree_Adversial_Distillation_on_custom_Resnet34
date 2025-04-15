import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class GeneratorA(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, img_size=32):
        super(GeneratorA, self).__init__()

        self.init_size = img_size//4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf*2*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False) 
        )

    def forward(self, z):
        out = self.l1(z.view(z.shape[0],-1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img



class GeneratorB(nn.Module):
    """ Generator from DCGAN: https://arxiv.org/abs/1511.06434
    """
    def __init__(self, nz=256, ngf=224, nc=3, img_size=64, slope=0.2):
        super(GeneratorB, self).__init__()
        if isinstance(img_size, (list, tuple)):
            self.init_size = ( img_size[0]//16, img_size[1]//16 )
        else:    
            self.init_size = ( img_size // 16, img_size // 16)

        self.project = nn.Sequential(
            Flatten(),
            nn.Linear(nz, ngf*8*self.init_size[0]*self.init_size[1]),
        )

        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf*8),
            
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(slope, inplace=True),
            # 2x

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(slope, inplace=True),
            # 4x
            
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 8x

            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 16x

            nn.Conv2d(ngf, nc, 3,1,1),
            nn.Tanh(),
        )

        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.BatchNorm2d)):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, z):
        proj = self.project(z)
        proj = proj.view(proj.shape[0], -1, self.init_size[0], self.init_size[1])
        output = self.main(proj)
        return output



class GeneratorD(nn.Module):
    def __init__(self, nz=100, ngf=8, nc=3, base_img_size=32, final_img_size=224):
        super(GeneratorD, self).__init__()

        assert base_img_size % 4 == 0, "base_img_size should be divisible by 4"
        self.base_img_size = base_img_size
        self.final_img_size = final_img_size
        self.init_size = base_img_size // 4

        self.l1 = nn.Sequential(
            nn.Linear(nz, ngf * 2 * self.init_size * self.init_size)
        )

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf * 2)
        )

        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.final_block = nn.Sequential(
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh()
        )

        self._init_weights()

    def forward(self, z):
        out = self.l1(z.view(z.shape[0], -1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)  # (B, ngf*2, H/4, W/4)
        out = self.conv_blocks0(out)
        out = F.interpolate(out, scale_factor=2)  # to H/2
        out = self.conv_blocks1(out)
        out = F.interpolate(out, scale_factor=2)  # to H
        out = self.conv_blocks2(out)
        img = self.final_block(out)

        # Final upscale to 224 x 224
        img = torchvision.transforms.Resize(size=(self.final_img_size, self.final_img_size))(img)
        
        return img

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)


import torch
import torch.nn as nn

class GeneratorDeconv(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, final_img_size=224):
        super(GeneratorDeconv, self).__init__()

        self.init_size = final_img_size // 16  # Start from 14x14 if 224x224 is target

        self.project = nn.Sequential(
            nn.Linear(nz, ngf * 8 * self.init_size * self.init_size),
            nn.ReLU(True)
        )

        self.model = nn.Sequential(
            nn.BatchNorm2d(ngf * 8),
            
            # 14x14 → 28x28
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, stride=2, padding=1),  
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # 28x28 → 56x56
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, stride=2, padding=1),  
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # 56x56 → 112x112
            nn.ConvTranspose2d(ngf * 2, ngf, 4, stride=2, padding=1),  
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # 112x112 → 224x224
            nn.ConvTranspose2d(ngf, nc, 4, stride=2, padding=1),  
            nn.Tanh()
        )

        self._initialize_weights()

    def forward(self, z):
        z = z.view(z.size(0), -1)  # Make sure it's [B, nz]
        out = self.project(z)  # Pass through linear layer
        out = out.view(z.size(0), -1, self.init_size, self.init_size)  # Reshape to [B, C, H, W]
        img = self.model(out)
        return img

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
