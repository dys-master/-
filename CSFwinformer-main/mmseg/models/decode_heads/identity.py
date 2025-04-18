import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Linear, build_activation_layer
from mmcv.runner import BaseModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
@HEADS.register_module()
class identityHead(BaseDecodeHead):

    def __init__(self, **kwargs):
        super(identityHead, self).__init__(**kwargs)
        # self.conv=nn.Conv2d(2,2,1,1)
    def forward(self, inputs):
        # inputs=self.conv(inputs)
        return inputs