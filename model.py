'''
2020-9-17 caoxin Copyright
'''


import torch
import torch.nn as nn



class CNN(nn.Module):
    def __init__(self,img_size, num_classes):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            #torch.nn.Con2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True)
            nn.Conv2d(3, 32, 3, 1,1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1,1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(0.2),
            
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, 1,1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, num_classes, 3, 1, 1),
            nn.LeakyReLU(0.2)

        )
        self.avg_pool = nn.AvgPool2d(img_size // 8) #average-pooling能减小第一种误差，更多的保留图像的背景信息，max-pooling能减小第二种误差，更多的保留纹理信息
        #对输入数据做线性变换：y=Ax+b
        #in_features - 每个输入样本的大小
		#out_features - 每个输出样本的大小
        self.classifier = nn.Linear(num_classes, num_classes)
    
    def forward(self, x):
        feature = self.conv(x)
        # print("-->{}".format(feature.size()))
        flatten = self.avg_pool(feature).view(feature.size(0), -1)
        # print("-->{}".format(flatten.size()))

        output = self.classifier(flatten)
        return output, feature