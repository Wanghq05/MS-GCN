import math
import pdb
# from dgl.nn.pytorch import GATv2Conv
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
from einops import rearrange, repeat
class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super(EdgeConv, self).__init__()

        self.k = k

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, dim=4):  # N, C, T, V

        x = self.get_graph_feature(x, self.k)  # 提取图特征
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]

        return x

    def knn(self, x, k):#选取6个层次集中欧式距离最近的三个层次集

        inner = -2 * torch.matmul(x.transpose(2, 1),
                                  x)  # N, V, V  torch.matmul(x.transpose(2, 1), x)如（1，2）16个通道的第一个节点乘以16个通道的对应第二个节点
        xx = torch.sum(x ** 2, dim=1, keepdim=True)  # 将每个通道的关节点数据叠加起来
        pairwise_distance = - xx - inner - xx.transpose(2, 1)  # （16*25*25）其实是相当于等到每两个关节点之间的距离

        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # 找到每个视频上面5个距离最近的关节点 N, V, k
        return idx

    def get_graph_feature(self, x, k, idx=None):
        N, C, L = x.size()
        if idx is None:
            idx = self.knn(x, k=k)
        device = x.get_device()

        idx_base = torch.arange(0, N, device=device).view(-1, 1, 1) * L

        idx = idx + idx_base
        idx = idx.view(-1)

        x = rearrange(x, 'n c v -> n v c')
        feature = rearrange(x, 'n v c -> (n v) c')[idx, :]  # 取16个batch size 里面每个视频里面关节点最近的数据
        feature = feature.view(N, L, k, C)
        x = repeat(x, 'n v c -> n v k c', k=k)

        feature = torch.cat((feature - x, x), dim=3)
        feature = rearrange(feature, 'n v k c -> n c v k')

        return feature


class HGA(nn.Module):
    def __init__(self, in_channels, num_layers=6):
        super(HGA, self).__init__()

        self.num_layers = num_layers

        # groups = get_groups(dataset='NTU', CoM=CoM)
        #
        # for i, group in enumerate(groups):
        #     group = [i - 1 for i in group]
        #     groups[i] = group

        inter_channels = in_channels // 4

        # self.layers = [groups[i] + groups[i + 1] for i in range(len(groups) - 1)]
        self.layers = [[1, 0, 20,                                             26, 25, 45],
                       [0, 20, 12, 16, 2, 4, 8,                               25, 45, 37, 41, 27, 29, 33],
                       [12, 16, 2, 4, 8, 13, 17, 3, 5, 9,     3, 28,                37, 41, 27, 29, 33, 38, 42, 28, 30, 34],
                       [13, 17, 3, 5, 9, 14, 18, 6, 10,       3, 28,                38, 42, 28, 30, 34, 39, 43, 31, 35],
                       [14, 18, 6, 10, 15, 19, 7, 11,                        39, 43, 31, 35, 40, 44, 32, 36],
                       [15, 19, 7, 11, 21, 22, 23, 24,                      40, 44, 32, 36, 46, 47, 48, 49]]
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

        self.edge_conv = EdgeConv(inter_channels, inter_channels, k=3)

        self.aggregate = nn.Conv1d(inter_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # N C L T V -> N C L 1 1
        # x= x.transpose(2, 0, 1, 3, 4)
        N, C, L, T, V = x.size()
        x_t = x.max(dim=-2, keepdim=False)[0]  # N, C, L, V
        x_t = self.conv_down(x_t)  # N C/r L V r = 6

        x_sampled = []
        for i in range(self.num_layers):  # Representative Average Pooling (RAP)
            s_t = x_t[:, :, i, self.layers[i]]
            s_t = s_t.mean(dim=-1, keepdim=True)  # N, C/r, 1, 1
            x_sampled.append(s_t)
        x_sampled = torch.cat(x_sampled, dim=2)  # N, C/r, L

        att = self.edge_conv(x_sampled, dim=3)  # N, C/r, L
        att = self.aggregate(att).view(N, C, L, 1, 1)

        out = (x * self.sigmoid(att)).sum(dim=2, keepdim=False)

        return out
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)  # 卷积参数得初始化


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 residual_kernel_size=1):
        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)  # 4个列表，包含四个卷积后的tensor

        out = torch.cat(branch_outs, dim=1)  # dim=1，在第二维进行拼接
        out += res
        return out


class MSGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(MSGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
    def forward(self, x, A=None, alpha=1):
        # 将64帧的关节点坐标叠加求平均 T的维度没有了，
        # 这里的关节点坐标通道ru：8，卷积后将每个坐标的（x，y，z）数据叠加（x1,x2,x3）
        # 相当于将64帧的关节点信息，求平均得到一帧的关节点信息
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        # 提取一共50个顶点，每个顶点之间的分布距离（50*50）
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        # 将关节之间的连接，加入到每个关节点之间距离分布之中，体现关节之间的边连接,
        # alpha是1 x1用来动态调整邻接矩阵
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        # x1 = self.tanh(x1)
        # x3代表数据输入的高级特征，x1体现关节之间的距离分布和边连接，
        # 聚合后，[64,50]这一维度的数据表示如 [1,1]的数据表示第一帧50个关节点坐标乘以每个关节点到第一个关节点距离分布，再进行叠加
        # 主要是体现数据的整体分布特征
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        # ([16, 64, 64, 50])
        return x1


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x



class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True, att = True):#att=True
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.num_layers = 6
        self.convs = nn.ModuleList()  # 以列表的形式来保持多个子模块(CTR-GC*3)
        self.att = att
        self.hga= HGA(out_channels, num_layers=6)

        for i in range(self.num_subset):
            self.convs.append(MSGC(in_channels, out_channels))  # 3个CTR-GC模块 conv[0]，conv[1]，conv[2]

        if residual:  # ture
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))  # 这是将其变成科学习得邻接矩阵
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        # self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:  # ture
            A = self.PA # A是可学习的邻接矩阵
            A = A.sum(1)
        else:
            A = self.A.cuda(x.get_device())
        y = []
        for i in range(self.num_subset):  # num_subset=3
            z = self.convs[i](x, A[i], self.alpha)  # alpha常数
            # y = z + y if y is not None else z
            y.append(z)  # A[1]单位矩阵。自身连接，
        y = torch.stack(y, dim=2)
        y = self.hga(y)
        y = self.bn(y)  # 经过三个CTR-GCN后，关节点数据超过1，所以要进行bn
        y += self.down(x)  # 无效操作
        y = self.relu(y)  # 避免梯度爆炸，降低参数数量
        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5,
                 dilations=[1, 2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class Model(nn.Module):
    # def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
    #              drop_out=0, adaptive=True):
    def __init__(self, num_class=120, num_point=50, num_person=1, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25

        self.num_class = num_class  # 类别
        self.num_point = num_point  # 关节点
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)  # 维度150

        base_channel = 64
        self.l1 = TCN_GCN_unit(base_channel, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        # self.l21 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel * 2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel * 2, base_channel * 4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive)
        self.std = True
        if self.std:
            self.input_map = nn.Sequential(
                nn.Conv2d(in_channels, base_channel // 2, kernel_size=(1, 1)),
                nn.BatchNorm2d(base_channel // 2),
                nn.LeakyReLU(0.1),
            )
            self.diff_map1 = nn.Sequential(
                nn.Conv2d(in_channels, base_channel // 8, kernel_size=(1, 1)),
                nn.BatchNorm2d(base_channel // 8),
                nn.LeakyReLU(0.1),
            )
            self.diff_map2 = nn.Sequential(
                nn.Conv2d(in_channels, base_channel // 8, kernel_size=(1, 1)),
                nn.BatchNorm2d(base_channel // 8),
                nn.LeakyReLU(0.1),
            )
            self.diff_map3 = nn.Sequential(
                nn.Conv2d(in_channels, base_channel // 8, kernel_size=(1, 1)),
                nn.BatchNorm2d(base_channel // 8),
                nn.LeakyReLU(0.1),
            )
            self.diff_map4 = nn.Sequential(
                nn.Conv2d(3, base_channel // 8, kernel_size=(1, 1)),
                nn.BatchNorm2d(base_channel // 8),
                nn.LeakyReLU(0.1),
            )

        self.fc = nn.Linear(base_channel * 4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()  # [16, 3, 64, 50, 1]
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)  # （16，150，64）
        x = self.data_bn(x)  # 对关节点数据进行归一化处理
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)  # （16，3，64，50）
        if self.std:
            # short-term modeling
            dif1 = x[:, :, 1:, :] - x[:, :, 0:-1, :] #求的是每一帧之间的偏移量 # 16 3 63 50
            dif1 = torch.cat([dif1.new(N * M, C, 1, V).zero_(), dif1], dim=-2)
            dif2 = x[:, :, :-1, :] - x[:, :, 1:, :]#反过来了
            dif2 = torch.cat([dif2, dif2.new(N * M, C, 1, V).zero_()], dim=-2)
            dif3 = x[:, :, 2:, :] - x[:, :, 0:-2, :]  # 每两帧之间的偏移量16 3 62 50
            dif3 = torch.cat([dif3.new(N * M, C, 2, V).zero_(), dif3], dim=-2)
            dif4 = x[:, :, :-2, :] - x[:, :, 2:, :]#
            dif4 = torch.cat([dif4, dif4.new(N * M, C, 2, V).zero_()], dim=-2)
            x = torch.cat((self.input_map(x), self.diff_map1(dif1), self.diff_map2(dif2),
                           self.diff_map3(dif3), self.diff_map4(dif4)), dim=1)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
        x = self.fc(x)  #
        return x
