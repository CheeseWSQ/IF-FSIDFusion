import torch
import torch.nn as nn
import math
import os
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils.dataset import H5Dataset
from tensorboardX import SummaryWriter


# 实现随机深度技术
# 用于模型的前向传递，并通过提供一种形式的正则化来帮助防止训练期间的过拟合。
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    # 创建一个形状与“x”相同的随机张量（但第一个维度是批量大小，其余维度是1）
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    # 该张量填充有“keep_prob”（即“1-drop_prob”）和1之间的随机值。
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize二值化（小于1的值变为0，等于或大于1的值变成1）
    # 去掉了“x”的随机元素（将它们设置为0），并缩放其余元素以保持期望值
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AttentionBase(nn.Module):
    def __init__(self,
                 dim,   
                 num_heads=8,
                 qkv_bias=False,):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        # 通过两个卷积层（“self.qkv1”和“self.qcv2”）来计算查询、关键字和值（qkv），再分别计算attention需要的q，k，v
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out


# 实现深度卷积和门控机制
# 深度卷积分别对每个输入通道进行操作，在某些情况下比标准卷积更有效
# 门控机制允许模型控制信息流，这可以帮助它关注最相关的特征
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 ffn_expansion_factor = 2,
                 bias = False):
        super().__init__()
        hidden_features = int(in_features*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        # 该层的输出沿着通道维度被分成两个块（‘x1’和‘x2’）
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # 门控机制
        x = F.gelu(x1) * x2
        # 深度卷积
        x = self.project_out(x)
        return x


# 提取浅层信息（BFE）
class BaseFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,  
                 qkv_bias=False,):
        super(BaseFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor,)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# 反向残差块，这是MobileNetV2等高效模型中的关键组件
# 作用：提高深度神经网络的性能以及防止梯度消失的问题
class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)  # 计算块中隐藏层的尺寸
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),  # 升维度
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),  # 向输入张量添加填充的填充层
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),  # 降维度
            # nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)


# 细节保持变换允许模型在变换输入的细节信息的同时保持输入的细节
# 对应论文中提到的INN
class DetailNode(nn.Module):
    def __init__(self, inp=32, oup=32, expand_ratio=2):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=inp, oup=oup, expand_ratio=expand_ratio)
        self.theta_rho = InvertedResidualBlock(inp=inp, oup=oup, expand_ratio=expand_ratio)
        self.theta_eta = InvertedResidualBlock(inp=inp, oup=oup, expand_ratio=expand_ratio)
        self.shffleconv = nn.Conv2d(inp*2, oup*2, kernel_size=1,
                                    stride=1, padding=0, bias=True)
    def separateFeature(self, x):
        # z1获得x的前半数列（向下取整）剩下的给z2
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2
    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2

class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
    # 将输入分为两部分，每个部分单独处理的体系结构中。然后将结果进行组合（沿着维度1连接）以产生最终输出。
    def forward(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)


class MultiScaleDetailFeatureExtraction(nn.Module):
    def __init__(self, input_channel, output_channel, num_layers=3):
        super(MultiScaleDetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode(input_channel, output_channel) for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
    # 将输入分为两部分，每个部分单独处理的体系结构中。然后将结果进行组合（沿着维度1连接）以产生最终输出。
    def forward(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)

# =============================================================================

# =============================================================================
import numbers
##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


# 层规范化的变体，该变体不包括偏差项。独立对单独样本特征进行归一化
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


# 带偏置项的层规范化
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias  # 减去均值除以方差


# 选择是否需要偏置的层规范化
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
# Restormer 模型中的部件
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
# Restormer 模型中的部件
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(1, out_plane//4, kernel_size=3, stride=1, relu=True),  # 正常卷积
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),  # 降维度
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),  # 正常卷积
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),  # 降维度
            nn.InstanceNorm2d(out_plane, affine=True)  # 归一化并对齐角点像素
        )

    def forward(self, x):
        x = self.main(x)
        return x


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = LayerNorm(out_features, 'BiasFree')

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = F.relu(out)
        out = self.norm(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = LayerNorm(out_features, 'BiasFree')
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))  # 缩小一半

    def forward(self, x):
        out = self.pool(x)
        out = self.conv(out)
        out = F.relu(out)
        out = self.norm(out)    
        return out


class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


class Restormer_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 down_scale=False,
                 up_scale=False
                 ):

        super(Restormer_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads = heads[2])
        self.detailFeature = DetailFeatureExtraction()
             
    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        base_feature = self.baseFeature(out_enc_level1)
        detail_feature = self.detailFeature(out_enc_level1)
        return base_feature, detail_feature, out_enc_level1
    
##########2024.12.30加的开始##########
class SharedFeatureCross(nn.Module):
    def __init__(self,
                 inp_channel=64,
                 dim=64,
                 out_channel=64,
                 expand_ratio=2,
                 num_res=4,
                 num_heads=2):
        super(SharedFeatureCross, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=inp_channel, oup=out_channel, expand_ratio=expand_ratio)
        self.theta_rho = InvertedResidualBlock(inp=inp_channel, oup=out_channel, expand_ratio=expand_ratio)
        self.theta_eta = InvertedResidualBlock(inp=inp_channel, oup=out_channel, expand_ratio=expand_ratio)
        self.shffleconv = nn.Conv2d(inp_channel*2, out_channel*2, kernel_size=1,
                                    stride=1, padding=0, bias=True)
        
        self.Trans1  = BaseFeatureExtraction(dim=dim, num_heads = num_heads)
        self.Trans2  = BaseFeatureExtraction(dim=dim, num_heads = num_heads)
        self.Trans3  = BaseFeatureExtraction(dim=dim, num_heads = num_heads)
        self.Trans4  = BaseFeatureExtraction(dim=dim, num_heads = num_heads)
        self.AFF = AFF()

    def forward(self, fx, fy):
        fx = self.Trans1(fx)
        fy = self.Trans2(fy) + self.theta_phi(fx)
        fx = fx * torch.exp(self.theta_rho(fy)) + self.theta_eta(fy)
        
        shared_feature_x = self.Trans3(fx)
        shared_feature_y = self.Trans4(fy)
        shared_feature = self.AFF(shared_feature_x, shared_feature_y)

        return shared_feature 
    

class SpecificFeatureCross(nn.Module):
    def __init__(self,
                 inp_channel=64,
                 dim=64,
                 out_channel=64,
                 expand_ratio=2,
                 num_res=4,
                 num_heads=2):
        super(SpecificFeatureCross, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=inp_channel, oup=out_channel, expand_ratio=expand_ratio)
        self.theta_rho = InvertedResidualBlock(inp=inp_channel, oup=out_channel, expand_ratio=expand_ratio)
        self.theta_eta = InvertedResidualBlock(inp=inp_channel, oup=out_channel, expand_ratio=expand_ratio)
        self.shffleconv = nn.Conv2d(inp_channel*2, out_channel*2, kernel_size=1,
                                    stride=1, padding=0, bias=True)
        
        # self.Trans1  = BaseFeatureExtraction(dim=dim, num_heads = num_heads)
        # self.Trans2  = BaseFeatureExtraction(dim=dim, num_heads = num_heads)
        # self.Trans3  = BaseFeatureExtraction(dim=dim, num_heads = num_heads)
        # self.Trans4  = BaseFeatureExtraction(dim=dim, num_heads = num_heads)
        self.CNN1 = DetailFeatureExtraction()
        self.CNN2 = DetailFeatureExtraction()
        self.CNN3 = DetailFeatureExtraction()
        self.CNN4 = DetailFeatureExtraction()
        self.AFF = AFF()

    def forward(self, fx, fy):
        fx = self.CNN1(fx)
        fy = self.CNN2(fy) + self.theta_phi(fx)
        fx = fx * torch.exp(self.theta_rho(fy)) + self.theta_eta(fy)
        
        specific_feature_x = self.CNN3(fx)
        specific_feature_y = self.CNN4(fy)
        specific_feature = self.AFF(specific_feature_x, specific_feature_y)

        return specific_feature


class Restormer_resolve_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 down_scale=False,
                 up_scale=False
                 ):

        super(Restormer_resolve_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.baseFeature_vis = BaseFeatureExtraction(dim=dim, num_heads = heads[2])
        self.baseFeature_ir  = BaseFeatureExtraction(dim=dim, num_heads = heads[2])
        self.detailFeature_vis = DetailFeatureExtraction()
        self.detailFeature_ir  = DetailFeatureExtraction()
        self.duelFuse = SharedFeatureCross()

    def forward(self, inp_img_vis, inp_img_ir):
        # inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1_vis = self.encoder_level1(self.patch_embed(inp_img_vis))
        out_enc_level1_ir = self.encoder_level1(self.patch_embed(inp_img_ir))
        base_feature_vis = self.baseFeature_vis(out_enc_level1_vis)
        detail_feature_vis = self.detailFeature_vis(out_enc_level1_vis)
        base_feature_ir = self.baseFeature_ir(out_enc_level1_ir)
        detail_feature_ir = self.detailFeature_ir(out_enc_level1_ir)
        shared_feature = self.duelFuse(base_feature_ir, base_feature_vis)
        return shared_feature, detail_feature_vis, detail_feature_ir
    


##########2024.12.30加的结束##########


class Restormer_Decoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Restormer_Decoder, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)
        self.sigmoid = nn.Sigmoid()              
    def forward(self, inp_img, base_feature, detail_feature):
        out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
        out_enc_level0 = self.reduce_channel(out_enc_level0)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1), out_enc_level0


class MultiScale_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 down_scale=False,
                 up_scale=False
                 ):
        super(MultiScale_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim*2, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=dim*4, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.downsample_1 = DownBlock2d(dim, 2 * dim)
        self.downsample_2 = DownBlock2d(2 * dim, 4 * dim)
        self.SCM1 = SCM(dim*2)
        self.SCM2 = SCM(dim*4)
        self.FAM1 = FAM(dim*2)
        self.FAM2 = FAM(dim*4)
        # self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        # self.detailFeature = DetailFeatureExtraction()
        self.baseFeature_level1 = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        self.baseFeature_level2 = BaseFeatureExtraction(dim=dim*2, num_heads=heads[2])
        self.baseFeature_level3 = BaseFeatureExtraction(dim=dim*4, num_heads=heads[2])
        self.detailFeature_level1 = MultiScaleDetailFeatureExtraction(dim//2, dim//2)
        self.detailFeature_level2 = MultiScaleDetailFeatureExtraction(dim, dim)
        self.detailFeature_level3 = MultiScaleDetailFeatureExtraction(dim*2, dim*2)

    def forward(self, inp_img):
        inp_img_2 = F.interpolate(inp_img, scale_factor=0.5)
        inp_img_4 = F.interpolate(inp_img_2, scale_factor=0.5)
        inp_embed = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_embed)  # C*H*W inner feature
        inp_shallow_img_2 = self.SCM1(inp_img_2)
        inp_enc_level1 = inp_shallow_img_2 + self.downsample_1(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level1)  # 2C*H/2*W/2 inner feature
        inp_shallow_img_4 = self.SCM2(inp_img_4)
        inp_enc_level2 = inp_shallow_img_4 + self.downsample_2(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level2)  # 4C*H/4*W/4 inner feature

        base_feature_level1 = self.baseFeature_level1(out_enc_level1)
        base_feature_level2 = self.baseFeature_level2(out_enc_level2)
        base_feature_level3 = self.baseFeature_level3(out_enc_level3)
        detail_feature_level1 = self.detailFeature_level1(out_enc_level1)
        detail_feature_level2 = self.detailFeature_level2(out_enc_level2)
        detail_feature_level3 = self.detailFeature_level3(out_enc_level3)

        # base_feature = self.baseFeature(out_enc_level1)
        # detail_feature = self.detailFeature(out_enc_level1)
        # return base_feature, detail_feature, out_enc_level1
        return (base_feature_level1, base_feature_level2, base_feature_level3,
                detail_feature_level1, detail_feature_level2, detail_feature_level3)


class MultiScale_Decoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(MultiScale_Decoder, self).__init__()
        self.reduce_channel_level1 = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)
        self.reduce_channel_level2 = nn.Conv2d(int(dim * 4), int(dim*2), kernel_size=1, bias=bias)
        self.reduce_channel_level3 = nn.Conv2d(int(dim * 8), int(dim*4), kernel_size=1, bias=bias)

        self.multiscale_fusion_channel_level1 = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)
        self.multiscale_fusion_channel_level2 = nn.Conv2d(int(dim * 4), int(dim*2), kernel_size=1, bias=bias)
        self.multiscale_fusion_channel_level3 = nn.Conv2d(int(dim * 8), int(dim*4), kernel_size=1, bias=bias)

        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(dim=dim*2, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level3 = nn.Sequential(
            *[TransformerBlock(dim=dim*4, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.upsample_1 = UpBlock2d(dim*2, dim)
        self.upsample_2 = UpBlock2d(dim*4, dim*2)

        self.output_level1 = nn.Sequential(
            nn.Conv2d(int(dim), int(dim) // 2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim) // 2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias), )
        self.output_level2 = nn.Sequential(
            nn.Conv2d(int(dim*2), int(dim*2) // 2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim*2) // 2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias), )
        self.output_level3 = nn.Sequential(
            nn.Conv2d(int(dim*4), int(dim*4) // 2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim*4) // 2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias), )
        self.sigmoid_level1 = nn.Sigmoid()
        self.sigmoid_level2 = nn.Sigmoid()
        self.sigmoid_level3 = nn.Sigmoid()

    def forward(self, inp_img, base_feature_level1, base_feature_level2, base_feature_level3,
                detail_feature_level1, detail_feature_level2, detail_feature_level3):
        # out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
        # out_enc_level0 = self.reduce_channel(out_enc_level0)
        # out_enc_level1 = self.encoder_level2(out_enc_level0)

        inp_img_2 = F.interpolate(inp_img, scale_factor=0.5)
        inp_img_4 = F.interpolate(inp_img_2, scale_factor=0.5)

        cat_enc_level1 = torch.cat((base_feature_level1, detail_feature_level1), dim=1)  #  128*H*W
        cat_enc_level2 = torch.cat((base_feature_level2, detail_feature_level2), dim=1)  #  256*H*W
        cat_enc_level3 = torch.cat((base_feature_level3, detail_feature_level3), dim=1)  #  512*H*W

        red_enc_level1 = self.reduce_channel_level1(cat_enc_level1)  # 64*H*W
        red_enc_level2 = self.reduce_channel_level2(cat_enc_level2)  # 128*H*W
        red_enc_level3 = self.reduce_channel_level3(cat_enc_level3)  # 256*H*W

        out_enc_level3 = self.encoder_level3(red_enc_level3)  # output 送到 output layer中
        inp_enc_level2 = torch.cat((self.upsample_2(out_enc_level3), red_enc_level2), dim=1)
        inp_enc_level2 = self.multiscale_fusion_channel_level2(inp_enc_level2)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level1 = torch.cat((self.upsample_1(out_enc_level2), red_enc_level1), dim=1)
        inp_enc_level1 = self.multiscale_fusion_channel_level1(inp_enc_level1)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        if inp_img is not None:
            out_enc_level3 = self.output_level3(out_enc_level3) + inp_img_4
            out_enc_level2 = self.output_level2(out_enc_level2) + inp_img_2
            out_enc_level1 = self.output_level1(out_enc_level1) + inp_img
        else:
            out_enc_level3 = self.output_level3(out_enc_level3)
            out_enc_level2 = self.output_level2(out_enc_level2)
            out_enc_level1 = self.output_level1(out_enc_level1)

        return (self.sigmoid_level1(out_enc_level1), self.sigmoid_level2(out_enc_level2), self.sigmoid_level3(out_enc_level3),
                out_enc_level1, out_enc_level2, out_enc_level3)


class MultiScale_inner_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 down_scale=False,
                 up_scale=False
                 ):
        super(MultiScale_inner_Encoder, self).__init__()

        self.patch_embed1 = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed2 = OverlapPatchEmbed(dim, dim*2)
        self.patch_embed3 = OverlapPatchEmbed(dim*2, dim*4)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim*2, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=dim*4, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.downsample_1 = DownBlock2d(dim, 2 * dim)
        self.downsample_2 = DownBlock2d(2 * dim, 4 * dim)

        # self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        # self.detailFeature = DetailFeatureExtraction()
        self.baseFeature_level1 = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        self.baseFeature_level2 = BaseFeatureExtraction(dim=dim*2, num_heads=heads[2])
        self.baseFeature_level3 = BaseFeatureExtraction(dim=dim*4, num_heads=heads[2])
        self.detailFeature_level1 = MultiScaleDetailFeatureExtraction(dim//2, dim//2)
        self.detailFeature_level2 = MultiScaleDetailFeatureExtraction(dim, dim)
        self.detailFeature_level3 = MultiScaleDetailFeatureExtraction(dim*2, dim*2)

    def forward(self, inp_img):

        inp_embed = self.patch_embed1(inp_img)
        out_enc_level1 = self.encoder_level1(inp_embed)  # C*H*W inner feature
        inp_enc_level1 = self.patch_embed2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level1)  # 2C*H*W inner feature
        inp_enc_level2 = self.patch_embed3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level2)  # 4C*H*W inner feature

        base_feature_level1 = self.baseFeature_level1(out_enc_level1)
        base_feature_level2 = self.baseFeature_level2(out_enc_level2)
        base_feature_level3 = self.baseFeature_level3(out_enc_level3)
        detail_feature_level1 = self.detailFeature_level1(out_enc_level1)
        detail_feature_level2 = self.detailFeature_level2(out_enc_level2)
        detail_feature_level3 = self.detailFeature_level3(out_enc_level3)

        # base_feature = self.baseFeature(out_enc_level1)
        # detail_feature = self.detailFeature(out_enc_level1)
        # return base_feature, detail_feature, out_enc_level1
        return (base_feature_level1, base_feature_level2, base_feature_level3,
                detail_feature_level1, detail_feature_level2, detail_feature_level3)


class MultiScale_inner_Decoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(MultiScale_inner_Decoder, self).__init__()
        self.reduce_channel_level1 = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)
        self.reduce_channel_level2 = nn.Conv2d(int(dim * 4), int(dim*2), kernel_size=1, bias=bias)
        self.reduce_channel_level3 = nn.Conv2d(int(dim * 8), int(dim*4), kernel_size=1, bias=bias)

        self.multiscale_fusion_channel_level1 = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)
        self.multiscale_fusion_channel_level2 = nn.Conv2d(int(dim * 4), int(dim*2), kernel_size=1, bias=bias)
        self.multiscale_fusion_channel_level3 = nn.Conv2d(int(dim * 8), int(dim*4), kernel_size=1, bias=bias)

        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(dim=dim*2, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level3 = nn.Sequential(
            *[TransformerBlock(dim=dim*4, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.upsample_1 = UpBlock2d(dim*2, dim)
        self.upsample_2 = UpBlock2d(dim*4, dim*2)

        self.output_level1 = nn.Sequential(
            nn.Conv2d(int(dim), int(dim) // 2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim) // 2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias), )
        self.output_level2 = nn.Sequential(
            nn.Conv2d(int(dim*2), int(dim*2) // 2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim*2) // 2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias), )
        self.output_level3 = nn.Sequential(
            nn.Conv2d(int(dim*4), int(dim*4) // 2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim*4) // 2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias), )
        self.sigmoid_level1 = nn.Sigmoid()
        # self.sigmoid_level2 = nn.Sigmoid()
        # self.sigmoid_level3 = nn.Sigmoid()

    def forward(self, inp_img, base_feature_level1, base_feature_level2, base_feature_level3,
                detail_feature_level1, detail_feature_level2, detail_feature_level3):
        # out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
        # out_enc_level0 = self.reduce_channel(out_enc_level0)
        # out_enc_level1 = self.encoder_level2(out_enc_level0)

        cat_enc_level1 = torch.cat((base_feature_level1, detail_feature_level1), dim=1)  #  128*H*W
        cat_enc_level2 = torch.cat((base_feature_level2, detail_feature_level2), dim=1)  #  256*H*W
        cat_enc_level3 = torch.cat((base_feature_level3, detail_feature_level3), dim=1)  #  512*H*W

        red_enc_level1 = self.reduce_channel_level1(cat_enc_level1)  # 64*H*W
        red_enc_level2 = self.reduce_channel_level2(cat_enc_level2)  # 128*H*W
        red_enc_level3 = self.reduce_channel_level3(cat_enc_level3)  # 256*H*W

        out_enc_level3 = self.encoder_level3(red_enc_level3)  # output 送到 output layer中
        inp_enc_level2 = torch.cat((self.upsample_2(out_enc_level3), red_enc_level2), dim=1)
        inp_enc_level2 = self.multiscale_fusion_channel_level2(inp_enc_level2)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level1 = torch.cat((self.upsample_1(out_enc_level2), red_enc_level1), dim=1)
        inp_enc_level1 = self.multiscale_fusion_channel_level1(inp_enc_level1)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        if inp_img is not None:
            out_enc_level1 = self.output_level1(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output_level1(out_enc_level1)

        return (self.sigmoid_level1(out_enc_level1), out_enc_level1)

class MultiscaleBaseFusion(nn.Module):
    def __init__(self, dim=64, num_heads=8):
        super(MultiscaleBaseFusion, self).__init__()
        self.base_fusion_level1 = BaseFeatureExtraction(dim=dim, num_heads=num_heads)
        self.base_fusion_level2 = BaseFeatureExtraction(dim=dim*2, num_heads=num_heads)
        self.base_fusion_level3 = BaseFeatureExtraction(dim=dim*4, num_heads=num_heads)

    def forward(self, base_feature_1, base_feature_2, base_feature_3):
        base_feature_fusion1 = self.base_fusion_level1(base_feature_1)
        base_feature_fusion2 = self.base_fusion_level2(base_feature_2)
        base_feature_fusion3 = self.base_fusion_level3(base_feature_3)
        return base_feature_fusion1, base_feature_fusion2, base_feature_fusion3


class MultiscaleDetailFusion(nn.Module):
    def __init__(self, dim=64):
        super(MultiscaleDetailFusion, self).__init__()
        self.detail_fusion_level1 = MultiScaleDetailFeatureExtraction(dim//2, dim//2)
        self.detail_fusion_level2 = MultiScaleDetailFeatureExtraction(dim, dim)
        self.detail_fusion_level3 = MultiScaleDetailFeatureExtraction(dim*2, dim*2)

    def forward(self, detail_feature_1, detail_feature_2, detail_feature_3):
        detail_feature_fusion_1 = self.detail_fusion_level1(detail_feature_1)
        detail_feature_fusion_2 = self.detail_fusion_level2(detail_feature_2)
        detail_feature_fusion_3 = self.detail_fusion_level3(detail_feature_3)
        return detail_feature_fusion_1, detail_feature_fusion_2, detail_feature_fusion_3


class MultiscaleBaseFusion_AFF(nn.Module):
    def __init__(self, dim=64, num_heads=8):
        super(MultiscaleBaseFusion_AFF, self).__init__()
        self.base_fusion_level1 = BaseFeatureExtraction(dim=dim, num_heads=num_heads)
        self.base_fusion_level2 = BaseFeatureExtraction(dim=dim*2, num_heads=num_heads)
        self.base_fusion_level3 = BaseFeatureExtraction(dim=dim*4, num_heads=num_heads)
        self.aff_fusion_level1 = AFF(channels=dim)
        self.aff_fusion_level2 = AFF(channels=dim*2)
        self.aff_fusion_level3 = AFF(channels=dim*4)

    def forward(self, base_feature_x1, base_feature_y1, base_feature_x2, base_feature_y2, base_feature_x3, base_feature_y3):
        base_feature_fusion1 = self.base_fusion_level1(self.aff_fusion_level1(base_feature_x1, base_feature_y1))
        base_feature_fusion2 = self.base_fusion_level2(self.aff_fusion_level2(base_feature_x2, base_feature_y2))
        base_feature_fusion3 = self.base_fusion_level3(self.aff_fusion_level3(base_feature_x3, base_feature_y3))
        return base_feature_fusion1, base_feature_fusion2, base_feature_fusion3


class MultiscaleDetailFusion_AFF(nn.Module):
    def __init__(self, dim=64, num_layers=1):
        super(MultiscaleDetailFusion_AFF, self).__init__()
        self.detail_fusion_level1 = MultiScaleDetailFeatureExtraction(dim//2, dim//2, num_layers=num_layers)
        self.detail_fusion_level2 = MultiScaleDetailFeatureExtraction(dim, dim, num_layers=num_layers)
        self.detail_fusion_level3 = MultiScaleDetailFeatureExtraction(dim*2, dim*2, num_layers=num_layers)
        self.aff_fusion_level1 = AFF(channels=dim)
        self.aff_fusion_level2 = AFF(channels=dim*2)
        self.aff_fusion_level3 = AFF(channels=dim*4)

    def forward(self, detail_feature_x1, detail_feature_y1, detail_feature_x2, detail_feature_y2, detail_feature_x3, detail_feature_y3):
        detail_feature_fusion_1 = self.detail_fusion_level1(self.aff_fusion_level1(detail_feature_x1, detail_feature_y1))
        detail_feature_fusion_2 = self.detail_fusion_level2(self.aff_fusion_level2(detail_feature_x2, detail_feature_y2))
        detail_feature_fusion_3 = self.detail_fusion_level3(self.aff_fusion_level3(detail_feature_x3, detail_feature_y3))
        return detail_feature_fusion_1, detail_feature_fusion_2, detail_feature_fusion_3


class Encoder_Block(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(Encoder_Block, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.sharedfeature_enc = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                                bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.basefeature_enc = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        self.detailfeature_enc = MultiScaleDetailFeatureExtraction(dim//2, dim//2)

    def forward(self, inp_img):
        inp_embed = self.patch_embed(inp_img)
        shared_feature = self.sharedfeature_enc(inp_embed)
        base_feature = self.basefeature_enc(shared_feature)
        detail_feature = self.detailfeature_enc(shared_feature)
        return shared_feature, base_feature, detail_feature


class Decoder_Block(nn.Module):
    def __init__(self,
                 inp_channels=256,
                 out_channels=128,
                 dim=256,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(Decoder_Block, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.embed1 = OverlapPatchEmbed(dim, dim)
        self.embed2 = OverlapPatchEmbed(dim, out_channels)
        self.decoder = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, feature1, feature2, inp_img):
        if feature2 is None:
            feature_embed = self.embed1(feature1)
            out_enc_level1 = self.decoder(feature_embed)
            out_enc_embed = self.embed2(out_enc_level1)
            if inp_img is not None:
                out_enc_level1 = self.output(out_enc_level1) + inp_img
            else:
                out_enc_level1 = self.output(out_enc_level1)
            
        else:
            feature = torch.cat((feature1, feature2), dim=1)
            feature = self.reduce_channel(feature)
            feature_embed = self.embed1(feature)
            out_enc_level1 = self.decoder(feature_embed)
            out_enc_embed = self.embed2(out_enc_level1)
            if inp_img is not None:
                out_enc_level1 = self.output(out_enc_level1) + inp_img
            else:
                out_enc_level1 = self.output(out_enc_level1)

        return self.sigmoid(out_enc_level1), out_enc_embed 

class MCDD_Encoder(nn.Module):
    def __init__(self,
                 inp_channel=1,
                 dim=[64,128,256],
                 ):
        super(MCDD_Encoder, self).__init__()
        self.encoder_level1 = Encoder_Block(inp_channels=inp_channel, dim=dim[0])
        self.encoder_level2 = Encoder_Block(inp_channels=dim[0], dim=dim[1])
        self.encoder_level3 = Encoder_Block(inp_channels=dim[1], dim=dim[2])

    def forward(self, inp_img):
        shared_feature1, base_feature1, detail_feature1 = self.encoder_level1(inp_img)
        shared_feature2, base_feature2, detail_feature2 = self.encoder_level2(shared_feature1)
        shared_feature3, base_feature3, detail_feature3 = self.encoder_level3(shared_feature2)
        return base_feature1, base_feature2, base_feature3, detail_feature1, detail_feature2, detail_feature3


class MCDD_Decoder(nn.Module):
    def __init__(self,
                 inp_channel=256,
                 dim=[128,64,1],
                 bias=False,
                 ):
        super(MCDD_Decoder, self).__init__()
        self.decoder_level3 = Decoder_Block(inp_channels=inp_channel, out_channels=dim[0], dim=inp_channel)
        self.decoder_level2 = Decoder_Block(inp_channels=dim[0], out_channels=dim[1], dim=dim[0])
        self.decoder_level1 = Decoder_Block(inp_channels=dim[1], out_channels=dim[2], dim=dim[1])
        self.reduce_channel3 = nn.Conv2d(int(inp_channel*2), int(inp_channel), kernel_size=1, bias=bias)
        self.reduce_channel2 = nn.Conv2d(int(dim[0]*2), int(dim[0]), kernel_size=1, bias=bias)
        self.reduce_channel1 = nn.Conv2d(int(dim[1]*2), int(dim[1]), kernel_size=1, bias=bias)

    def forward(self, inp_img, base_feature1, base_feature2, base_feature3, detail_feature1, detail_feature2, detail_feature3):
        feature_level3 = self.reduce_channel3(torch.cat((base_feature3, detail_feature3), dim=1))
        _, out_feature_level3 = self.decoder_level3(feature_level3, None, inp_img)

        feature_level2 = self.reduce_channel2(torch.cat((base_feature2, detail_feature2), dim=1))
        _, out_feature_level2 = self.decoder_level2(out_feature_level3, feature_level2, inp_img)

        feature_level1 = self.reduce_channel1(torch.cat((base_feature1, detail_feature1), dim=1))     
        out_img, out_feture_level1 = self.decoder_level1(out_feature_level2, feature_level1, inp_img)

        return out_img, out_feture_level1


class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True):
        super(Cross_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        # self.qkv_dwconv = nn.Conv2d(
        #     dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.qkv_vis = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv_vis = nn.Conv2d(
             dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.qkv_ir = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv_ir = nn.Conv2d(
             dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, ir, vis):
        b, c, h, w = vis.shape

        # qkv = self.qkv_dwconv(self.qkv(x))
        # q, k, v = qkv.chunk(3, dim=1)
        
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)',
        #               head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)',
        #               head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)',
        #               head=self.num_heads)
        qkv_vis = self.qkv_dwconv_vis(self.qkv_vis(vis))
        qkv_ir =self.qkv_dwconv_ir(self.qkv_ir(ir))
        q_vis, k_vis, v_vis = qkv_vis.chunk(3, dim=1)
        q_ir, k_ir, v_ir = qkv_ir.chunk(3, dim=1)
        
        q_vis = rearrange(q_vis, 'b (head c) h w -> b head c (h w)',
                          head=self.num_heads)
        k_vis = rearrange(k_vis, 'b (head c) h w -> b head c (h w)',
                          head=self.num_heads)
        v_vis = rearrange(v_vis, 'b (head c) h w -> b head c (h w)',
                          head=self.num_heads)
        
        q_ir = rearrange(q_ir, 'b (head c) h w -> b head c (h w)',
                         head=self.num_heads)
        k_ir = rearrange(k_ir, 'b (head c) h w -> b head c (h w)',
                         head=self.num_heads)
        v_ir = rearrange(v_ir, 'b (head c) h w -> b head c (h w)',
                         head=self.num_heads)

        q_vis = torch.nn.functional.normalize(q_vis, dim=-1)
        k_ir = torch.nn.functional.normalize(k_ir, dim=-1)

        # attn = (q @ k.transpose(-2, -1)) * self.temperature
        # attn = attn.softmax(dim=-1)
        attn = (q_vis @ k_ir.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # out = (attn @ v)
        # out = rearrange(out, 'b head c (h w) -> b (head c) h w',
        #                 head=self.num_heads, h=h, w=w)
        # out = self.project_out(out)
        out_vis = (attn @ v_vis)
        out_ir = (attn @ v_ir)

        out_vis = rearrange(out_vis, 'b head c (h w) -> b (head c) h w',
                            head=self.num_heads, h=h, w=w)
        out_ir = rearrange(out_ir, 'b head c (h w) -> b (head c) h w',
                           head=self.num_heads, h=h, w=w)

        out_vis = self.project_out(out_vis)
        out_ir = self.project_out(out_ir)

        return out_vis, out_ir



class Cross_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 down_scale=False,
                 up_scale=False
                 ):
        super(Cross_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.downsample_base1 = DownBlock2d(dim, 2 * dim)
        self.downsample_base2 = DownBlock2d(2 * dim, 4 * dim)
        self.downsample_detail1 = DownBlock2d(dim, 2 * dim)
        self.downsample_detail2 = DownBlock2d(2 * dim, 4 * dim)
        self.upsample_base1 = UpBlock2d(2 * dim, dim)
        self.upsample_base2 = UpBlock2d(4 * dim, 2 * dim)
        self.upsample_detail1 = UpBlock2d(2 * dim, dim)
        self.upsample_detail2 = UpBlock2d(4 * dim, 2 * dim)
        self.upsample_cross_base1 = UpBlock2d(2 * dim, dim)
        self.upsample_cross_base2 = UpBlock2d(4 * dim, 2 * dim)
        self.upsample_cross_detail1 = UpBlock2d(2 * dim, dim)
        self.upsample_cross_detail2 = UpBlock2d(4 * dim, 2 * dim)

        self.cross_attn = Cross_Attention(4 * dim)       

        # self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        # self.detailFeature = DetailFeatureExtraction()
        self.baseFeature_level1 = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        self.baseFeature_level2 = BaseFeatureExtraction(dim=dim*2, num_heads=heads[2])
        self.baseFeature_level3 = BaseFeatureExtraction(dim=dim*4, num_heads=heads[2])
        self.baseFeature_level4 = BaseFeatureExtraction(dim=dim*2, num_heads=heads[2])
        self.baseFeature_level5 = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        self.detailFeature_level1 = MultiScaleDetailFeatureExtraction(dim//2, dim//2)
        self.detailFeature_level2 = MultiScaleDetailFeatureExtraction(dim, dim)
        self.detailFeature_level3 = MultiScaleDetailFeatureExtraction(dim*2, dim*2)
        self.detailFeature_level4 = MultiScaleDetailFeatureExtraction(dim, dim)
        self.detailFeature_level5 = MultiScaleDetailFeatureExtraction(dim//2, dim//2)

        self.out_base = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=1)
        self.out_detail = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=1)

    def forward(self, inp_img):
        # shared feature
        inp_embed = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level(inp_embed)  # C*H*W inner feature
        # base featre and detail feature
        base_feature_level1 = self.baseFeature_level1(out_enc_level1)
        inp_base_12 = self.downsample_base1(base_feature_level1)
        detail_feature_level1 = self.detailFeature_level1(out_enc_level1)
        inp_detail_12 = self.downsample_detail1(detail_feature_level1)

        base_feature_level2 = self.baseFeature_level2(inp_base_12) # 128*64*64 
        inp_base_23 = self.downsample_base2(base_feature_level2)
        detail_feature_level2 = self.detailFeature_level2(inp_detail_12)
        inp_detail_23 = self.downsample_detail2(detail_feature_level2)

        base_feature_cross3, detail_feature_cross3 = self.cross_attn(inp_base_23, inp_detail_23) # 256*32*32 
        base_feature_level3 = self.baseFeature_level3(inp_base_23) + base_feature_cross3       
        detail_feature_level3 = self.detailFeature_level3(inp_detail_23) - detail_feature_cross3

        base_feature_cross4 = self.upsample_cross_base2(base_feature_cross3) # 
        detail_feature_cross4 = self.upsample_cross_detail2(detail_feature_cross3)
        inp_base_34 = self.upsample_base2(base_feature_level3)
        inp_detail_34 = self.upsample_detail2(detail_feature_level3)
        base_feature_level4 = self.baseFeature_level4(inp_base_34) + base_feature_cross4
        detail_feature_level4 = self.detailFeature_level4(inp_detail_34) - detail_feature_cross4

        base_feature_cross5 = self.upsample_cross_base1(base_feature_cross4)
        detail_feature_cross5 = self.upsample_cross_detail1(detail_feature_cross4)
        inp_base_45 = self.upsample_base1(base_feature_level4)
        inp_detail_45 = self.upsample_detail1(detail_feature_level4)
        base_feature_level5 = self.baseFeature_level5(inp_base_45) + base_feature_cross5
        detail_feature_level5 = self.detailFeature_level5(inp_detail_45) - detail_feature_cross5

        out_base_feature = self.out_base(base_feature_level5)
        out_detail_feature = self.out_detail(detail_feature_level5)
        return out_base_feature, out_detail_feature
        # base_feature = self.baseFeature(out_enc_level1)
        # detail_feature = self.detailFeature(out_enc_level1)
        # return base_feature, detail_feature, out_enc_level1
        # return (base_feature_level1, base_feature_level2, base_feature_level3,
        #         detail_feature_level1, detail_feature_level2, detail_feature_level3)


class Cross_Modality_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 down_scale=False,
                 up_scale=False
                 ):
        super(Cross_Modality_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.downsample_base1 = DownBlock2d(dim, 2 * dim)
        self.downsample_base2 = DownBlock2d(2 * dim, 4 * dim)
        self.downsample_detail1 = DownBlock2d(dim, 2 * dim)
        self.downsample_detail2 = DownBlock2d(2 * dim, 4 * dim)
        self.upsample_base1 = UpBlock2d(2 * dim, dim)
        self.upsample_base2 = UpBlock2d(4 * dim, 2 * dim)
        self.upsample_detail1 = UpBlock2d(2 * dim, dim)
        self.upsample_detail2 = UpBlock2d(4 * dim, 2 * dim)
        self.upsample_cross_base1 = UpBlock2d(2 * dim, dim)
        self.upsample_cross_base2 = UpBlock2d(4 * dim, 2 * dim)
        self.upsample_cross_detail1 = UpBlock2d(2 * dim, dim)
        self.upsample_cross_detail2 = UpBlock2d(4 * dim, 2 * dim)

        self.cross_attn = Cross_Attention(4 * dim)       

        # self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        # self.detailFeature = DetailFeatureExtraction()
        self.baseFeature_level1 = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        self.baseFeature_level2 = BaseFeatureExtraction(dim=dim*2, num_heads=heads[2])
        self.baseFeature_level3 = BaseFeatureExtraction(dim=dim*4, num_heads=heads[2])
        self.baseFeature_level4 = BaseFeatureExtraction(dim=dim*2, num_heads=heads[2])
        self.baseFeature_level5 = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        self.detailFeature_level1 = MultiScaleDetailFeatureExtraction(dim//2, dim//2)
        self.detailFeature_level2 = MultiScaleDetailFeatureExtraction(dim, dim)
        self.detailFeature_level3 = MultiScaleDetailFeatureExtraction(dim*2, dim*2)
        self.detailFeature_level4 = MultiScaleDetailFeatureExtraction(dim, dim)
        self.detailFeature_level5 = MultiScaleDetailFeatureExtraction(dim//2, dim//2)

        self.out_base = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=1)
        self.out_detail = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=1)

    def forward(self, inp_img):
        # shared feature
        inp_embed = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level(inp_embed)  # C*H*W inner feature
        # base featre and detail feature
        base_feature_level1 = self.baseFeature_level1(out_enc_level1)
        inp_base_12 = self.downsample_base1(base_feature_level1)
        detail_feature_level1 = self.detailFeature_level1(out_enc_level1)
        inp_detail_12 = self.downsample_detail1(detail_feature_level1)

        base_feature_level2 = self.baseFeature_level2(inp_base_12) # 128*64*64 
        inp_base_23 = self.downsample_base2(base_feature_level2)
        detail_feature_level2 = self.detailFeature_level2(inp_detail_12)
        inp_detail_23 = self.downsample_detail2(detail_feature_level2)

        base_feature_cross3, detail_feature_cross3 = self.cross_attn(inp_base_23, inp_detail_23) # 256*32*32 
        base_feature_level3 = self.baseFeature_level3(inp_base_23) + base_feature_cross3       
        detail_feature_level3 = self.detailFeature_level3(inp_detail_23) - detail_feature_cross3

        base_feature_cross4 = self.upsample_cross_base2(base_feature_cross3) # 
        detail_feature_cross4 = self.upsample_cross_detail2(detail_feature_cross3)
        inp_base_34 = self.upsample_base2(base_feature_level3)
        inp_detail_34 = self.upsample_detail2(detail_feature_level3)
        base_feature_level4 = self.baseFeature_level4(inp_base_34) + base_feature_cross4
        detail_feature_level4 = self.detailFeature_level4(inp_detail_34) - detail_feature_cross4

        base_feature_cross5 = self.upsample_cross_base1(base_feature_cross4)
        detail_feature_cross5 = self.upsample_cross_detail1(detail_feature_cross4)
        inp_base_45 = self.upsample_base1(base_feature_level4)
        inp_detail_45 = self.upsample_detail1(detail_feature_level4)
        base_feature_level5 = self.baseFeature_level5(inp_base_45) + base_feature_cross5
        detail_feature_level5 = self.detailFeature_level5(inp_detail_45) - detail_feature_cross5

        out_base_feature = self.out_base(base_feature_level5)
        out_detail_feature = self.out_detail(detail_feature_level5)
        return out_base_feature, out_detail_feature



if __name__ == '__main__':
    # height = 128
    # width = 128
    # window_size = 8
    # # modelE = Restormer_Encoder().cuda()
    # # modelD = Restormer_Decoder().cuda()
    # batch_size = 8
    # trainloader = DataLoader(H5Dataset(r"MSRS_train/MSRS_train_imgsize_128_stride_200.h5"),
    #                          batch_size=batch_size,
    #                          shuffle=True,
    #                          num_workers=0)
    # writer = SummaryWriter("logs")
    # model = BaseFeatureExtraction(dim=64, num_heads=8)
    #
    # loader = {'train': trainloader, }
    # data = torch.rand([8, 3, 128, 128])
    # # writer.add_graph(model, data)
    # with SummaryWriter(comment='LeNet') as w:
    #     w.add_graph(model, data)

    height = 128
    width = 128
    window_size = 8
    batch_size = 8
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = torch.rand([8, 1, 32, 32])
    encoder = nn.DataParallel(MCDD_Encoder()).to(device)
    decoder = nn.DataParallel(MCDD_Decoder()).to(device)
    (base_feature_level1, base_feature_level2, base_feature_level3,
     detail_feature_level1, detail_feature_level2, detail_feature_level3) = encoder(data)
    print(base_feature_level1.shape)
    print(base_feature_level2.shape)
    print(base_feature_level3.shape)
    print(detail_feature_level1.shape)
    print(detail_feature_level2.shape)
    print(detail_feature_level3.shape)
    rebuild_img, _ = decoder(data, base_feature_level1, base_feature_level2, base_feature_level3,
     detail_feature_level1, detail_feature_level2, detail_feature_level3)
    print(rebuild_img.shape)
    # print(rebuild_img_level2.shape)
    # print(rebuild_img_level3.shape)







