# ---------------------------------------------------------------
# Copyright (c) 2021, Nota AI GmbH. All rights reserved.
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule
from ..utils import resize
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import *
import math
from timm.models.layers import DropPath, trunc_normal_
"""
在前向传播阶段，该头部模块首先通过多个注意力块和MLP模块处理输入特征，然后使用不同的池化比例进行汇聚，最后通过线性层进行预测。
"""
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
# 对多个输入进行汇聚
class CatKey(nn.Module):
    def __init__(self, pool_ratio=[1,2,4,8], dim=[256,160,64,32]):
        super().__init__()
        self.pool_ratio = pool_ratio
        # 创建包含1x1卷积层的模块列表，用于对具有空间缩小的张量进行处理
        self.sr_list = nn.ModuleList([nn.Conv2d(dim[i], dim[i], kernel_size=1, stride=1)
                                      for i in range(len(self.pool_ratio))
                                      if self.pool_ratio[i] > 1])
        # 创建包含平均池化层的模块列表，用于对具有空间缩小的张量进行处理
        self.pool_list = nn.ModuleList([nn.AvgPool2d(self.pool_ratio[i], self.pool_ratio[i], ceil_mode=True)
                                        for i in range(len(self.pool_ratio))
                                        if self.pool_ratio[i] > 1])
#CatKey 模块接收一个包含多个张量的列表 x，通过对其中的每个张量进行池化和卷积，然后在通道维度上进行拼接。
    def forward(self, x):
        out_list = []
        cnt = 0
        # 遍历输入张量列表中的张量
        for i in range(len(self.pool_ratio)):
            if self.pool_ratio[i] > 1:
                # 对张量应用1x1卷积，然后进行平均池化
                out_list.append(self.sr_list[cnt](self.pool_list[cnt](x[i])))
                cnt += 1
            else:
                #  如果pool_ratio为1，则不执行任何操作，并将原始张量添加到列表中
                out_list.append(x[i])
        return torch.cat(out_list, dim=1)#(batch_size, channels, height, width)
#包含了一个跨注意力模块，通过使用线性层、深度可分离卷积层等组件实现。
class CrossAttention(nn.Module):
    def __init__(self, dim1, dim2, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_ratio=16):
        super().__init__()
        '''
        dim1：表示输入x的特征维度（query的维度）。
        dim2：表示输入y的特征维度（key和value的维度）。
        num_heads：表示注意力头的数量。
        qkv_bias：是否在线性变换中使用偏置。
        qk_scale：缩放因子，如果为None，则默认为头维度的平方根的倒数。
        attn_drop：注意力分数的Dropout率。
        proj_drop：投影层的Dropout率。
        pool_ratio：池化的比率。如果为负值，表示不使用池化。
        '''
        assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."

        self.dim1 = dim1
        self.dim2 = dim2
        self.num_heads = num_heads
        head_dim = dim1 // num_heads
        self.pool_ratio = pool_ratio
        self.scale = qk_scale or head_dim ** -0.5
        '''
        self.q：对query进行线性变换。
        self.kv：对key和value进行线性变换。
        self.attn_drop：注意力分数的Dropout。
        self.proj：对注意力输出进行线性变换。
        self.proj_drop：投影层的Dropout。
        如果pool_ratio >= 0，则会执行以下操作：

        self.pool1：对query进行平均池化。
        self.pool2：对key和value进行平均池化。
        self.sr1：对query进行卷积。
        self.sr2：对key和value进行卷积。
        最后，self.norm1和self.norm2分别是对query和key进行LayerNorm操作，self.act是GELU激活函数。

        初始化权重的过程在self.apply(self._init_weights)中完成，对于线性变换和卷积操作，采用了截断正态分布初始化的方式。
        '''
        self.q = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.kv = nn.Linear(dim2, dim1 * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.pool_ratio >= 0:
            self.pool1 = nn.AvgPool2d(2, 2) #query
            self.pool2 = nn.AvgPool2d(pool_ratio * 2, pool_ratio * 2) #key&value
            self.sr1 = nn.Conv2d(dim1, dim1, kernel_size=1, stride=1)
            self.sr2 = nn.Conv2d(dim2, dim2, kernel_size=1, stride=1)
        self.norm1 = nn.LayerNorm(dim1)
        self.norm2 = nn.LayerNorm(dim2)
        self.act = nn.GELU()
        self.apply(self._init_weights)
    #用于初始化模块的权重。对于线性变换和卷积操作，采用了截断正态分布初始化的方式
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y, H2, W2, H1, W1):
        '''
        q = self.q(x).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)：
        对查询张量x进行线性变换，得到注意力矩阵q。
        reshape 操作将结果重塑为四维张量，其中 B1 是批次大小，N1 是序列长度，self.num_heads 是头数，C1 // self.num_heads 是每个头的维度。
        permute 操作重新排列维度，使其变为 (B1, num_heads, N1, C1 // num_heads)。
        x_ = x.permute(0, 2, 1).reshape(B1, C1, H1, W1)：
        对输入张量x进行维度变换，将其从 (B1, N1, C1) 变为 (B1, C1, H1, W1)。
        x_ = self.sr1(self.pool1(x_)).reshape(B1, C1, -1).permute(0, 2, 1)：
        使用 pool1 对 x_ 进行平均池化，然后通过 sr1 进行卷积，对空间尺寸进行缩小。
        将结果通过 reshape 变为 (B1, C1, -1) 的形状，即 (B1, C1, H1 * W1 // (2 * 2))。
        最后通过 permute 将维度重新排列。
        x_ = self.norm1(x_)：
        对 x_ 进行Layer Normalization。
        x_ = self.act(x_)：
        应用GELU激活函数。
        N1 = N1 // (2 * 2)：
        更新序列长度，将其缩小为原来的1/4。
        q = self.q(x_).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)：
        通过对经过降维处理的 x_ 进行线性变换，得到新的注意力矩阵 q。
        y_ = self.norm2(y)：
        对输入张量 y 进行Layer Normalization。
        y_ = self.act(y_)：
        应用GELU激活函数。
        kv = self.kv(y_).reshape(B1, -1, 2, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4)：
        对经过处理的 y_ 进行线性变换，得到键值对 kv。
        reshape 操作将结果变为五维张量，其中维度的顺序是 (2, B1, num_heads, -1, C1 // num_heads)。
        attn = (q @ k.transpose(-2, -1)) * self.scale：
        计算注意力分数，这里使用了缩放操作 self.scale。
        attn = attn.softmax(dim=-1)：
        对注意力分数进行Softmax操作。
        attn = self.attn_drop(attn)：
        对注意力分数进行Dropout操作。
        x = (attn @ v).transpose(1, 2).reshape(B1, N1, C1)：
        计算加权和，得到输出张量 x。
        '''
        B1, N1, C1 = x.shape
        B2, N2, C2 = y.shape
        q = self.q(x).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)#对查询张量x进行线性变换，得到注意力矩阵q
        # torch.Size([1, 8, 1024, 64])
        x_ = x.permute(0, 2, 1).reshape(B1, C1, H1, W1)# 对输入张量x进行维度变换，将其从 (B1, N1, C1) 变为 (B1, C1, H1, W1)。
        x_ = self.sr1(self.pool1(x_)).reshape(B1, C1, -1).permute(0, 2, 1)#使用 pool1 对 x_ 进行平均池化，然后通过 sr1 进行卷积，对空间尺寸进行缩小将结果通过 reshape 变为 (B1, C1, -1) 的形状，即 (B1, C1, H1 * W1 // (2 * 2))。最后通过 permute 将维度重新排列
        x_ = self.norm1(x_)
        x_ = self.act(x_)#应用GELU激活函数
        N1 = N1 // (2 * 2)#更新序列长度，将其缩小为原来的1/4
        q = self.q(x_).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)#通过对经过降维处理的 x_ 进行线性变换，得到新的注意力矩阵 q
        # torch.Size([1, 8, 256, 64])
        # y_ = y.permute(0, 2, 1).reshape(B2, C2, H2, W2)
        # y_ = self.sr2(self.pool2(y_)).reshape(B2, C2, -1).permute(0, 2, 1)
        # y_ = self.norm2(y_)
        y_ = self.norm2(y)
        y_ = self.act(y_)
        kv = self.kv(y_).reshape(B1, -1, 2, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4)#对经过处理的 y_ 进行线性变换，得到键值对 kv, reshape 操作将结果变为五维张量，其中维度的顺序是 (2, B1, num_heads, -1, C1 // num_heads)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B1, N1, C1)

        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(1, 2).view(B1, C1, H1 // 2, W1 // 2)
        x = resize(x, size=(H1, W1), mode='bilinear', align_corners=False)
        x = x.flatten(2).transpose(1, 2)

        return x

class Block(nn.Module):
    """
    Block 模块进行多头注意力机制的处理。
    其中，Block 是一个通用的多头注意力模块，接收两个特征图 dim1 和 dim2，
    以及其他一些注意力机制的超参数。最终得到经过注意力处理后的特征图。
    """
    def __init__(self, dim1, dim2, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_ratio=16):
        super().__init__()
        self.norm1 = norm_layer(dim1)
        self.norm2 = norm_layer(dim2)
        self.norm3 = norm_layer(dim1)

        self.attn = CrossAttention(dim1=dim1, dim2=dim2, num_heads=num_heads, pool_ratio=pool_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y, H2, W2, H1, W1):
        x = self.norm1(x)
        y = self.norm2(y)
        x = x + self.drop_path(self.attn(x, y, H2, W2, H1, W1)) #self.norm2(y)이 F1에 대한 값
        x = self.norm3(x)
        x = x + self.drop_path(self.mlp(x, H1, W1))

        # x = x + self.drop_path(self.attn(self.norm1(x), self.norm2(y), H2, W2)) #self.norm2(y)이 F1에 대한 값
        # x = x + self.drop_path(self.mlp(self.norm3(x), H1, W1))

        return x

@MODELS.register_module()
class APFormerHeadCity(BaseDecodeHead):
    """
    Attention-Pooling Former
    """
    def __init__(self, feature_strides, pool_scales=(1, 2, 3, 6), **kwargs):
        super(APFormerHeadCity, self).__init__(input_transform='multiple_select', **kwargs)
        """设置了模块的各种属性和子模块。
        其中包括输入特征图的步长 feature_strides，
        不同层级的通道数 c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels，
        总通道数 tot_channels，
        以及注意力模块所需的参数（如嵌入维度 embedding_dim，注意头的数量 num_heads，池化比例 pool_ratio 等）。
        此外，初始化了一系列注意力模块 self.attn_c4, self.attn_c3, self.attn_c2, self.attn_c1，
        以及连接注意力模块的 self.cat_key1, self.cat_key2, self.cat_key3, self.cat_key4。
        最后，定义了两个卷积模块 self.linear_fuse 和 self.linear_pred。
        """

        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        tot_channels = sum(self.in_channels)# 为in_channels=[32, 64, 160, 256]之和=512,因为把四个阶段的输出拼接起来了,所以kv的维度一直为512

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        num_heads = decoder_params['num_heads']
        pool_ratio = decoder_params['pool_ratio']

        self.attn_c4 = Block(dim1=c4_in_channels, dim2=tot_channels, num_heads=num_heads[0], mlp_ratio=4,
                                drop_path=0.1, pool_ratio=8)
        self.attn_c3 = Block(dim1=c3_in_channels, dim2=tot_channels, num_heads=num_heads[1], mlp_ratio=4,
                                drop_path=0.1, pool_ratio=4)
        self.attn_c2 = Block(dim1=c2_in_channels, dim2=tot_channels, num_heads=num_heads[2], mlp_ratio=4,
                                drop_path=0.1, pool_ratio=2)
        self.attn_c1 = Block(dim1=c1_in_channels, dim2=tot_channels, num_heads=num_heads[3], mlp_ratio=4,
                                drop_path=0.1, pool_ratio=1)

        pool_ratio = [i * 2 for i in pool_ratio]
        """
        通过提供不同尺度的通道数 [c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels] 和对应的池化比率 pool_ratio，
        pool_ratio=[1,2,4,8], dim=[256,160,64,32]
        模块将执行以下操作：
        对具有不同通道数的输入张量进行池化和卷积操作。
        使用池化比率大于1的池化操作对相应的输入张量进行降维。
        对降维后的张量应用 1x1 卷积操作。
        """
        self.cat_key1 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key2 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key3 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key4 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])

        self.linear_fuse = ConvModule(
            in_channels=tot_channels,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        #对输入进行预处理，通过 _transform_inputs 方法，将输入特征图进行一些操作，返回列表 x，其中包含多个特征图，对应不同的尺度。
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        """
        当massroad数据集以1024大小,batchsize为1输入时,1为batchsize,in_channels=[32, 64, 160, 256],在基础config里的解码头设置该参数
        c1的大小为torch.Size([1, 32, 256, 256])
        c2的大小为torch.Size([1, 64, 128, 128])
        c3的大小为torch.Size([1, 160, 64, 64])
        c4的大小为torch.Size([1, 256, 32, 32])
        """
        c1, c2, c3, c4 = x
        ############## MLP decoder on C1-C4 ###########
        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        c_key = self.cat_key1([c4, c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2) #在多头注意力中，对输入张量进行线性变换后，需要在通道维度上分割成多个头，因此需要调整张量的维度shape: [batch, h1*w1, channels]
        #对1/32尺度的特征图进行多头注意力机制处理
        c4 = c4.flatten(2).transpose(1, 2)
        _c4 = self.attn_c4(c4, c_key, h4, w4, h4, w4)#c4, c_key的高度和宽度相等

        _c4 = _c4.permute(0,2,1).reshape(n, -1, h4, w4)#_c4，其形状为 [batch, channels, h4, w4]
        c_key = self.cat_key2([_c4, c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2) #shape: [batch, h1*w1, channels]
        c3 = c3.flatten(2).transpose(1, 2)
        _c3 = self.attn_c3(c3, c_key, h4, w4, h3, w3)#不同的层级有注意力机制应用在不同的尺寸上

        _c3 = _c3.permute(0,2,1).reshape(n, -1, h3, w3)
        c_key = self.cat_key3([_c4, _c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2) #shape: [batch, h1*w1, channels]
        c2 = c2.flatten(2).transpose(1, 2)
        _c2 = self.attn_c2(c2, c_key, h4, w4, h2, w2)

        _c2 = _c2.permute(0,2,1).reshape(n, -1, h2, w2)
        c_key = self.cat_key4([_c4, _c3, _c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2) #shape: [batch, h1*w1, channels]
        c1 = c1.flatten(2).transpose(1, 2)
        _c1 = self.attn_c1(c1, c_key, h4, w4, h1, w1)
        #尺度调整
        _c4 = resize(_c4, size=(h1,w1), mode='bilinear', align_corners=False)
        _c3 = resize(_c3, size=(h1,w1), mode='bilinear', align_corners=False)
        _c2 = resize(_c2, size=(h1,w1), mode='bilinear', align_corners=False)
        _c1 = _c1.permute(0,2,1).reshape(n, -1, h1, w1)
        #特征融合
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))#对四个特征图在通道维度上进行拼接，形成一个新的特征图

        x = self.dropout(_c)
        #最终预测
        x = self.linear_pred(x)#输出形状为 [batch, num_classes, h1, w1]

        return x
