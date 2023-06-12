import torch
import torch.nn as nn
import torch.nn.functional as F

from registry import MODEL_REGISTRY


class ResidualBlock(nn.Module):

    def __init__(self,inchannel,outchannel,stride=1,shortcut=None):

        super(ResidualBlock,self).__init__()

        self.left=nn.Sequential(
            nn.Conv2d(inchannel,outchannel,3,stride,1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,3,1,1,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right=shortcut

    def forward(self,x):

        out=self.left(x)
        residual=x if self.right is None else self.right(x)
        out+=residual
        return F.relu(out)

@MODEL_REGISTRY.register()
class Color_ResNet(nn.Module):


    def __init__(self,num_classes=3):

        super(Color_ResNet,self).__init__()

        self.pre=nn.Sequential(
            nn.Conv2d(2,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(3,2,1)
        )


        self.layer1=self._make_layer(64,64,3,stride=2)
        self.layer2=self._make_layer(64,128,2,stride=2)
        self.layer3=self._make_layer(128,256,2,stride=2)
        self.layer4=self._make_layer(256,512,3,stride=2)



        self.conv14 = nn.Conv2d(512, 1024, 7, stride=1, padding=3)
        nn.init.xavier_uniform_(self.conv14.weight)
        nn.init.constant_(self.conv14.bias, 0.1)
        # self.conv14_bn = nn.BatchNorm2d(4096)

        self.conv15 = nn.Conv2d(1024, 512, 1, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv15.weight)
        nn.init.constant_(self.conv15.bias, 0.1)
        # self.conv15_bn = nn.BatchNorm2d(512)

        self.upsampconv1 = nn.ConvTranspose2d(512, 512, 2, stride=2, padding=0)

        self.conv16 = nn.Conv2d(768, 512, 5, stride=1, padding=2)
        nn.init.xavier_uniform_(self.conv16.weight)
        nn.init.constant_(self.conv16.bias, 0.1)
        # self.conv16_bn = nn.BatchNorm2d(512)

        self.upsampconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2, padding=0)
        self.conv17 = nn.Conv2d(384, 256, 5, stride=1, padding=2)
        nn.init.xavier_uniform_(self.conv17.weight)
        nn.init.constant_(self.conv17.bias, 0.1)
        # self.conv17_bn = nn.BatchNorm2d(256)

        self.upsampconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0)
        self.conv18 = nn.Conv2d(192, 128, 5, stride=1, padding=2)
        nn.init.xavier_uniform_(self.conv18.weight)
        nn.init.constant_(self.conv18.bias, 0.1)
        # self.conv18_bn = nn.BatchNorm2d(128)

        self.upsampconv4 = nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0)
        self.conv19 = nn.Conv2d(128, 64, 5, stride=1, padding=2)
        nn.init.xavier_uniform_(self.conv19.weight)
        nn.init.constant_(self.conv19.bias, 0.1)
        # self.conv19_bn = nn.BatchNorm2d(64)

        self.upsampconv5 = nn.ConvTranspose2d(64, 32, 2, stride=2, padding=0)
        self.conv20 = nn.Conv2d(34, 64, 5, stride=1, padding=2)
        nn.init.xavier_uniform_(self.conv20.weight)
        nn.init.constant_(self.conv20.bias, 0.1)
        # self.conv20_bn = nn.BatchNorm2d(32)

        self.conv21 = nn.Conv2d(64, 2, 5, stride=1, padding=2)
        nn.init.xavier_uniform_(self.conv21.weight)
        nn.init.constant_(self.conv21.bias, 0.1)
    def _make_layer(self,inchannel,outchannel,bloch_num,stride=1):

        shortcut=nn.Sequential(
            nn.Conv2d(inchannel,outchannel,1,stride,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers=[]
        layers.append(ResidualBlock(inchannel,outchannel,stride,shortcut))
        for i in range(1,bloch_num):
            layers.append(ResidualBlock(outchannel,outchannel))
        return nn.Sequential(*layers)

    def forward(self,x):

        x1=self.pre(x)
        #print(x1.shape)#256
        x2=self.layer1(x1)
        #print(x2.shape)#128
        x3=self.layer2(x2)
        #print(x3.shape)#64
        x4=self.layer3(x3)
        #print(x4.shape)#32
        x5=self.layer4(x4)
        #print(x5.shape)#16


        x60 = F.relu(self.conv14(x5))

        #print(x.shape,'\n')

        x61 = F.relu(self.conv15(x60))

        #print(x.shape,'\n')

        x71 = F.relu(self.upsampconv1(x61))#32*32
        # if x71.size()[2]!=x4.size()[2] or x71.size()[3]!=x4.size()[3]:
          # x71 = nn.Upsample(size=(x4.size()[2],x4.size()[3]), mode='bilinear')(x71)
        x71 = nn.Upsample(size=(x4.size()[2],x4.size()[3]), mode='bilinear')(x71)

        x72 = torch.cat((x71, x4), 1)

        x73 = F.relu(self.conv16(x72))

        #print(x.shape,'\n')

        x81 = F.relu(self.upsampconv2(x73))#64*64
        # if x81.size()[2]!=x3.size()[2] or x81.size()[3]!=x3.size()[3]:
          # x81 = nn.Upsample(size=(x3.size()[2],x3.size()[3]), mode='bilinear')(x81)
        x81 = nn.Upsample(size=(x3.size()[2],x3.size()[3]), mode='bilinear')(x81)
        x82 = torch.cat((x81, x3), 1)
        x83 = F.relu(self.conv17(x82))

       # print(x.shape,'\n')


        x91 = F.relu(self.upsampconv3(x83))#128*128
        # if x91.size()[2]!=x2.size()[2] or x91.size()[3]!=x2.size()[3]:
          # x91 = nn.Upsample(size=(x2.size()[2],x2.size()[3]), mode='bilinear')(x91)
        x91 = nn.Upsample(size=(x2.size()[2],x2.size()[3]), mode='bilinear')(x91)
        x92 = torch.cat((x91, x2), 1)
        x93 = F.relu(self.conv18(x92))

        x101 = F.relu(self.upsampconv4(x93))#256*256
        # if x101.size()[2]!=x1.size()[2] or x101.size()[3]!=x1.size()[3]:
          # x101 = nn.Upsample(size=(x1.size()[2],x1.size()[3]), mode='bilinear')(x101)
        x101 = nn.Upsample(size=(x1.size()[2],x1.size()[3]), mode='bilinear')(x101)
        x102 = torch.cat((x101, x1), 1)
        x103 = F.relu(self.conv19(x102))

        x111 = F.relu(self.upsampconv5(x103))#512*512
        # if x111.size()[2]!=x.size()[2] or x111.size()[3]!=x.size()[3]:
          # x111 = nn.Upsample(size=(x.size()[2],x.size()[3]), mode='bilinear')(x111)
        x111 = nn.Upsample(size=(x.size()[2],x.size()[3]), mode='bilinear')(x111)
        x112 = torch.cat((x111, x), 1)
        x113 = F.relu(self.conv20(x112))

        x114 = torch.sigmoid(self.conv21(x113))
        #print(x114.shape)


        return x114



