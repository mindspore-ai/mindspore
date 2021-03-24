"""Bottleneck"""

import mindspore.nn as nn
from mindspore.ops import operations as P

class FPN(nn.Cell):
    """FPN"""
    def __init__(self, config, backbone, is_training=True):
        super(FPN, self).__init__()

        self.backbone = backbone
        feature_size = config.feature_size
        self.P5_1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, pad_mode='same')
        self.P_upsample1 = P.ResizeNearestNeighbor((feature_size[1], feature_size[1]))
        self.P5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, pad_mode='same')

        self.P4_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, pad_mode='same')
        self.P_upsample2 = P.ResizeNearestNeighbor((feature_size[0], feature_size[0]))
        self.P4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, pad_mode='same')

        self.P3_1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, pad_mode='same')
        self.P3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, pad_mode='same')

        self.P6_0 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, pad_mode='same')

        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, pad_mode='same')

        self.is_training = is_training
        if not is_training:
            self.activation = P.Sigmoid()

    def construct(self, x):
        """construct"""
        C3, C4, C5 = self.backbone(x)

        P5 = self.P5_1(C5)
        P5_upsampled = self.P_upsample1(P5)
        P5 = self.P5_2(P5)

        P4 = self.P4_1(C4)
        P4 = P5_upsampled + P4
        P4_upsampled = self.P_upsample2(P4)
        P4 = self.P4_2(P4)

        P3 = self.P3_1(C3)
        P3 = P4_upsampled + P3
        P3 = self.P3_2(P3)

        P6 = self.P6_0(C5)

        P7 = self.P7_1(P6)
        P7 = self.P7_2(P7)
        multi_feature = (P3, P4, P5, P6, P7)

        return multi_feature
        