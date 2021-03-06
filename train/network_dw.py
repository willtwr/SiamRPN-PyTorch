import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchvision.models import alexnet
from torch.autograd import Variable
from torch import nn

from config import config


class SiameseAlexNet(nn.Module):
    def __init__(self, ):
        super(SiameseAlexNet, self).__init__()
        self.featureExtract = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=2),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, 5),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3),
            nn.BatchNorm2d(256),
        )
        self.anchor_num = config.anchor_num
        self.input_size = config.detection_img_size
        self.score_displacement = int((self.input_size - config.template_img_size) / config.total_stride)
        
        self.conv_cls1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.conv_cls2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        
        self.conv_r1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.conv_r2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        
        self.cls_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2 * self.anchor_num, kernel_size=1, stride=1, padding=0),
            )
        self.regress_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4 * self.anchor_num, kernel_size=1, stride=1, padding=0),
            )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, std= 0.0005)
                nn.init.normal_(m.bias.data, std= 0.0005)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, template, detection):
        N = template.size(0)
        template_feature = self.featureExtract(template)
        detection_feature = self.featureExtract(detection)

        kernel_score = self.conv_cls1(template_feature)
        kernel_regression = self.conv_r1(template_feature)
        conv_score = self.conv_cls2(detection_feature)
        conv_regression = self.conv_r2(detection_feature)
        
        conv_scores = conv_score.reshape(1, -1, self.score_displacement + 6, self.score_displacement + 6)
        score_filters = kernel_score.reshape(-1, 1, 6, 6)
        pred_score = F.conv2d(conv_scores, score_filters, groups=N*256).reshape(N, 256, self.score_displacement + 1,
                                                                                self.score_displacement + 1)
        pred_score = self.cls_head(pred_score)

        conv_reg = conv_regression.reshape(1, -1, self.score_displacement + 6, self.score_displacement + 6)
        reg_filters = kernel_regression.reshape(-1, 1, 6, 6)
        pred_regression = F.conv2d(conv_reg, reg_filters, groups=N*256).reshape(N, 256, self.score_displacement + 1, 
                                                                                self.score_displacement + 1)
        pred_regression = self.regress_head(pred_regression)
        return pred_score, pred_regression

    def track_init(self, template):
        N = template.size(0)
        template_feature = self.featureExtract(template)

        kernel_score = self.conv_cls1(template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        kernel_regression = self.conv_r1(template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        self.score_filters = kernel_score.reshape(-1, 256, 4, 4)
        self.reg_filters = kernel_regression.reshape(-1, 256, 4, 4)

    def track(self, detection):
        N = detection.size(0)
        detection_feature = self.featureExtract(detection)

        conv_score = self.conv_cls2(detection_feature)
        conv_regression = self.conv_r2(detection_feature)

        conv_scores = conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        pred_score = F.conv2d(conv_scores, self.score_filters, groups=N).reshape(N, 10, self.score_displacement + 1,
                                                                                 self.score_displacement + 1)
        conv_reg = conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        pred_regression = self.regress_adjust(
            F.conv2d(conv_reg, self.reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                                   self.score_displacement + 1))
        return pred_score, pred_regression
