import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
import scipy.sparse as sp
from torch.nn.functional import cosine_similarity


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


# 数据嵌入
class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.token_embed(x)
        x = self.norm(x)
        return x


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()  # (100,64)
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)  # (100,1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()  # (1,32)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1,100,64)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x).detach()


# 拉普拉斯嵌入
class LaplacianPE(nn.Module):
    def __init__(self, lape_dim, embed_dim):
        super().__init__()
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)  # 8->64

    def forward(self, lap_mx):
        lap_pos_enc = self.embedding_lap_pos_enc(lap_mx).unsqueeze(0).unsqueeze(0)
        return lap_pos_enc


# 输入嵌入 特征维、拉普拉斯、位置、空间
class DataEmbedding(nn.Module):
    def __init__(
            self, feature_dim, embed_dim, lape_dim, adj_mx, drop=0.,
            add_time_in_day=False, add_day_in_week=False, device=torch.device('cpu'), num_nodes=170
    ):
        super().__init__()

        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week

        self.device = device
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.value_embedding = TokenEmbedding(feature_dim, embed_dim)  # 特征维 1->64
        self.position_encoding = PositionalEncoding(embed_dim)  # (1,100,64)
        if self.add_time_in_day:
            self.minute_size = 1440
            self.daytime_embedding = nn.Embedding(self.minute_size, embed_dim)
            # 天嵌入 1440->64
        if self.add_day_in_week:
            weekday_size = 7
            self.weekday_embedding = nn.Embedding(weekday_size, embed_dim)
            # 周嵌入 7->64
        self.spatial_embedding = LaplacianPE(lape_dim, embed_dim)  # 拉普拉斯嵌入
        self.tempp_embedding = nn.Linear(lape_dim, embed_dim)  # 时间序列嵌入
        self.dropout = nn.Dropout(drop)

    def forward(self, x, lap_mx, tempp):
        origin_x = x
        x = self.value_embedding(origin_x[:, :, :, :self.feature_dim])  # self.feature_dim = 1
        x += self.position_encoding(x)
        if self.add_time_in_day:
            x += self.daytime_embedding((origin_x[:, :, :, self.feature_dim] * self.minute_size).round().long())
        if self.add_day_in_week:
            x += self.weekday_embedding(origin_x[:, :, :, self.feature_dim + 1: self.feature_dim + 8].argmax(dim=3))
        x += self.spatial_embedding(lap_mx)  # mask掉拉普拉斯嵌入
        x += self.tempp_embedding(tempp).unsqueeze(0).unsqueeze(0)
        x = self.dropout(x)
        return x


# 随机丢弃
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :x.shape[2] - self.chomp_size, :].contiguous()


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nwv->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

# 时空自注意力
class STSelfAttention(nn.Module):
    def __init__(
            self, dim, s_attn_size, t_attn_size, geo_num_heads=4, sem_num_heads=2, t_num_heads=2, qkv_bias=False,
            attn_drop=0., proj_drop=0., device=torch.device('cpu'), output_dim=1, num_nodes=170
    ):
        super().__init__()
        assert dim % (geo_num_heads + sem_num_heads + t_num_heads) == 0
        self.geo_num_heads = geo_num_heads  # 4
        self.sem_num_heads = sem_num_heads  # 2
        self.t_num_heads = t_num_heads  # 2
        self.head_dim = dim // (geo_num_heads + sem_num_heads + t_num_heads)  # 8
        self.scale = self.head_dim ** -0.5
        self.device = device
        self.s_attn_size = s_attn_size
        self.t_attn_size = t_attn_size
        self.geo_ratio = geo_num_heads / (geo_num_heads + sem_num_heads + t_num_heads)  # 0.5
        self.sem_ratio = sem_num_heads / (geo_num_heads + sem_num_heads + t_num_heads)  # 0.25
        self.t_ratio = 1 - self.geo_ratio - self.sem_ratio  # 0.25
        # dim 64
        self.output_dim = output_dim

        # self.pattern_q_linears = nn.ModuleList([
        #     nn.Linear(dim, int(dim * self.geo_ratio)) for _ in range(output_dim)
        # ])
        # self.pattern_k_linears = nn.ModuleList([
        #     nn.Linear(dim, int(dim * self.geo_ratio)) for _ in range(output_dim)
        # ])
        # self.pattern_v_linears = nn.ModuleList([
        #     nn.Linear(dim, int(dim * self.geo_ratio)) for _ in range(output_dim)
        # ])

        self.geo_q_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_k_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_v_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_attn_drop = nn.Dropout(attn_drop)

        # self.sem_q_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        # self.sem_k_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        # self.sem_v_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        # self.sem_attn_drop = nn.Dropout(attn_drop)

        self.t_q_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_k_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_v_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)

        # self.expand = nn.Linear(int(dim / 2), dim)
        self.proj = nn.Linear(int(dim * 3 / 4), dim)
        # self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 多图
        self.gconv = nn.ModuleList()
        self.days = 288
        dims = 40
        self.supports_len = 1
        torch.cuda.manual_seed_all(1)  # 为所有的GPU设置种子
        self.nodevec_p1 = nn.Parameter(torch.randn(self.days, dims).to(device), requires_grad=True).to(device)
        self.nodevec_p2 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec_p3 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec_pk = nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device)
        

        # for i in range(4):
        #     self.gconv.append(
        #         gcn(32, 32, 0.3, support_len=self.supports_len, order=2))
        self.gconv = gcn(32, 32, 0.3, support_len=self.supports_len, order=2)

        self.reshape1 = nn.Linear(dim, 32)
        self.reshape2 = nn.Linear(32, dim)

    def dgconstruct(self, time_embedding, source_embedding, target_embedding, core_embedding):
        # print("time type:", type(time_embedding))
        adp = torch.einsum('ai, ijk->ajk', time_embedding, core_embedding)
        adp = torch.einsum('bj, ajk->abk', source_embedding, adp)
        adp = torch.einsum('ck, abk->abc', target_embedding, adp)
        adp = F.relu(adp)
        adp = F.softmax(adp, dim=2)
        return adp
        

    def forward(self, x, ind, x_patterns, pattern_keys, geo_mask=None, sem_mask=None):
        B, T, N, D = x.shape  # 16,12,170,64

        # 动态图
        ind %= self.days
        ind = ind.cpu().numpy()
        adp = self.dgconstruct(self.nodevec_p1[ind], self.nodevec_p2, self.nodevec_p3, self.nodevec_pk)
        # adp 64,170,170
        new_supports = [adp]

        # 动态图卷积
        x = x.reshape(-1, D)
        x = self.reshape1(x)
        x = x.reshape(B, T, N, 32)
        x = x.permute(0, 3, 2, 1)  # 16,32,170,12
        # for i in range(4):
        #     x = self.gconv[i](x, new_supports)
        x = self.gconv(x, new_supports)
        x = x.permute(0, 3, 2, 1)  # 16,12,170,32
        x = x.reshape(-1, 32)
        x = self.reshape2(x)
        x = x.reshape(B, T, N, D)

        # 时间注意力
        t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_q = t_q.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale
        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)
        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T, int(D * self.t_ratio)).transpose(1, 2)

        

        # 近距离空间注意力
        geo_q = self.geo_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_k = self.geo_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # for i in range(self.output_dim):
        #     pattern_q = self.pattern_q_linears[i](x_patterns[..., i])  # x_patterns(16,12,170,64,1)
        #     pattern_k = self.pattern_k_linears[i](pattern_keys[..., i])  # pattern_keys(16,64,1)
        #     pattern_v = self.pattern_v_linears[i](pattern_keys[..., i])
        #     pattern_attn = (pattern_q @ pattern_k.transpose(-2, -1)) * self.scale
        #     pattern_attn = pattern_attn.softmax(dim=-1)
        #     geo_k += pattern_attn @ pattern_v
        geo_v = self.geo_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_q = geo_q.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_k = geo_k.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_v = geo_v.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_attn = (geo_q @ geo_k.transpose(-2, -1)) * self.scale
        if geo_mask is not None:
            geo_attn.masked_fill_(geo_mask, float('-inf'))
        geo_attn = geo_attn.softmax(dim=-1)
        geo_attn = self.geo_attn_drop(geo_attn)
        geo_x = (geo_attn @ geo_v).transpose(2, 3).reshape(B, T, N, int(D * self.geo_ratio))

        # 远距离空间注意力
        # sem_q = self.sem_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # sem_k = self.sem_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # sem_v = self.sem_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # sem_q = sem_q.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        # sem_k = sem_k.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        # sem_v = sem_v.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        # sem_attn = (sem_q @ sem_k.transpose(-2, -1)) * self.scale
        # if sem_mask is not None:
        #     sem_attn.masked_fill_(sem_mask, float('-inf'))
        # sem_attn = sem_attn.softmax(dim=-1)
        # sem_attn = self.sem_attn_drop(sem_attn)
        # sem_x = (sem_attn @ sem_v).transpose(2, 3).reshape(B, T, N, int(D * self.sem_ratio))

        # 合并
        # x = self.expand(geo_x)
        x = self.proj(torch.cat([t_x, geo_x], dim=-1))
        # x = self.proj(t_x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# 时间自注意力
class TemporalSelfAttention(nn.Module):
    def __init__(
            self, dim, dim_out, t_attn_size, t_num_heads=6, qkv_bias=False,
            attn_drop=0., proj_drop=0., device=torch.device('cpu'),
    ):
        super().__init__()
        assert dim % t_num_heads == 0
        self.t_num_heads = t_num_heads
        self.head_dim = dim // t_num_heads
        self.scale = self.head_dim ** -0.5
        self.device = device
        self.t_attn_size = t_attn_size

        self.t_q_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_k_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_v_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, N, D = x.shape
        t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_q = t_q.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale

        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)

        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T, D).transpose(1, 2)

        x = self.proj(t_x)
        x = self.proj_drop(x)
        return x


# 时空块
class STEncoderBlock(nn.Module):

    def __init__(
            self, dim, s_attn_size, t_attn_size, geo_num_heads=4, sem_num_heads=2, t_num_heads=2, mlp_ratio=4.,
            qkv_bias=True, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, device=torch.device('cpu'), type_ln="pre",
            output_dim=1, num_nodes=170
    ):
        super().__init__()
        self.type_ln = type_ln
        self.norm1 = norm_layer(dim)
        self.st_attn = STSelfAttention(
            dim, s_attn_size, t_attn_size, geo_num_heads=geo_num_heads, sem_num_heads=sem_num_heads,
            t_num_heads=t_num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, device=device, output_dim=output_dim, num_nodes=num_nodes
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, ind, x_patterns, pattern_keys, geo_mask=None, sem_mask=None):
        if self.type_ln == 'pre':  # ture
            x = x + self.drop_path(
                self.st_attn(self.norm1(x), ind, x_patterns, pattern_keys, geo_mask=geo_mask, sem_mask=sem_mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.type_ln == 'post':
            x = self.norm1(
                x + self.drop_path(self.st_attn(x, x_patterns, pattern_keys, geo_mask=geo_mask, sem_mask=sem_mask)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 针对tempp矩阵，取最大值
def norm_embedding(adj):
    torch.fill_(adj.diagonal(), 0)
    values, indices = torch.topk(adj, 5, dim=1)
    b = torch.zeros_like(adj)
    b.scatter_(1, indices, 1.0)
    return b

# 针对dtw矩阵，取最小值
# def norm_embedding(adj):
#     torch.fill_(adj.diagonal(), float('inf'))
#     values, indices = adj.topk(5, dim=1, largest=False)
#     b = torch.zeros_like(adj)
#     b.scatter_(1, indices, 1.0)
#     return b

class DGSTA(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get("num_nodes", 1)
        self.feature_dim = self.data_feature.get("feature_dim", 1)
        self.ext_dim = self.data_feature.get("ext_dim", 0)
        self.num_batches = self.data_feature.get('num_batches', 1)
        self.dtw_matrix = self.data_feature.get('dtw_matrix')
        self.adj_mx = data_feature.get('adj_mx')
        sd_mx = data_feature.get('sd_mx')
        sh_mx = data_feature.get('sh_mx')
        self._logger = getLogger()
        self.dataset = config.get('dataset')

        self.embed_dim = config.get('embed_dim', 64)
        self.skip_dim = config.get("skip_dim", 256)
        lape_dim = config.get('lape_dim', 8)  # 位置嵌入 8
        geo_num_heads = config.get('geo_num_heads', 4)
        sem_num_heads = config.get('sem_num_heads', 2)
        t_num_heads = config.get('t_num_heads', 2)
        mlp_ratio = config.get("mlp_ratio", 4)
        qkv_bias = config.get("qkv_bias", True)
        drop = config.get("drop", 0.)
        attn_drop = config.get("attn_drop", 0.)
        drop_path = config.get("drop_path", 0.3)
        self.s_attn_size = config.get("s_attn_size", 3)
        self.t_attn_size = config.get("t_attn_size", 3)
        enc_depth = config.get("enc_depth", 6)
        type_ln = config.get("type_ln", "pre")
        self.type_short_path = config.get("type_short_path", "hop")

        self.output_dim = config.get('output_dim', 1)
        self.input_window = config.get("input_window", 12)
        self.output_window = config.get('output_window', 12)
        add_time_in_day = config.get("add_time_in_day", True)
        add_day_in_week = config.get("add_day_in_week", True)
        self.device = config.get('device', torch.device('cpu'))
        self.world_size = config.get('world_size', 1)
        self.huber_delta = config.get('huber_delta', 1)
        self.quan_delta = config.get('quan_delta', 0.25)
        self.far_mask_delta = config.get('far_mask_delta', 5)  # geo矩阵参数 7
        self.dtw_delta = config.get('dtw_delta', 5)  # dtw参数 5

        self.use_curriculum_learning = config.get('use_curriculum_learning', True)
        self.step_size = config.get('step_size', 2500)  # 2776
        self.max_epoch = config.get('max_epoch', 200)
        self.task_level = config.get('task_level', 0)
        self.lape_dim = config.get('lape_dim', 200)  # 8
        # for i in range(self.adj_mx.shape[0]):
        #     for j in range(self.adj_mx.shape[1]):
        #         if self.adj_mx[i][j] == 0:
        #             self.adj_mx[i][j] = 1
        #         else:
        #             self.adj_mx[i][j] = 0
        # self.adj_mx = torch.from_numpy(self.adj_mx).to(self.device)

        # 这两行没看懂是干什么的，好像没用到
        # self.data_path = './raw_data/' + self.dataset + '/'
        # self.adp_file = config.get('adp_file', self.dataset)

        if self.max_epoch * self.num_batches * self.world_size < self.step_size * self.output_window:
            self._logger.warning('Parameter `step_size` is too big with {} epochs and '
                                 'the model cannot be trained for all time steps.'.format(self.max_epoch))
        if self.use_curriculum_learning:
            self._logger.info('Use use_curriculum_learning!')

        if self.type_short_path == "dist":  # false
            distances = sd_mx[~np.isinf(sd_mx)].flatten()
            std = distances.std()
            sd_mx = np.exp(-np.square(sd_mx / std))
            self.far_mask = torch.zeros(self.num_nodes, self.num_nodes).to(self.device)
            self.far_mask[sd_mx < self.far_mask_delta] = 1
            self.far_mask = self.far_mask.bool()
        else:
            sh_mx = sh_mx.T  # 转置
            # geo矩阵 距离近的为0，距离远的为1
            self.geo_mask = torch.zeros(self.num_nodes, self.num_nodes).to(self.device)
            self.geo_mask[sh_mx >= self.far_mask_delta] = 1  # sh_mx矩阵里大于7的设为1
            self.geo_mask = self.geo_mask.bool()
            # self.geo_mask = self.adj_mx.bool()
            # sem矩阵 距离近的为0，距离远的为1
            self.sem_mask = torch.ones(self.num_nodes, self.num_nodes).to(self.device)
            # (170,170) 全为1
            sem_mask = self.dtw_matrix.argsort(axis=1)[:, :self.dtw_delta]
            # 对dtw_matrix按行排序求出索引值，将每行前dtw_delta个索引值切割，赋值给sem_mask
            # (170,5)
            for i in range(self.sem_mask.shape[0]):
                self.sem_mask[i][sem_mask[i]] = 0  # 将每行的对应5处位置值设为0
            # self.sem_mask = torch.load("./libcity/cache/dataset_cache/dtw.npy").to(self.device)
            self.sem_mask = self.sem_mask.bool()

        # 对pattern_keys做一次全连接，3维扩到64维
        self.pattern_keys = torch.from_numpy(data_feature.get('pattern_keys')).float().to(self.device)
        self.pattern_embeddings = nn.ModuleList([
            TokenEmbedding(self.s_attn_size, self.embed_dim) for _ in range(self.output_dim)
        ])

        # 对输入进行拉普拉斯、时间、位置嵌入
        self.enc_embed_layer = DataEmbedding(
            self.feature_dim - self.ext_dim, self.embed_dim, lape_dim, self.adj_mx, drop=drop,
            add_time_in_day=add_time_in_day, add_day_in_week=add_day_in_week, device=self.device, num_nodes=self.num_nodes
        )

        #  返回0~0.3，总个数为6的等差数列
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path, enc_depth)]

        #  时空编码
        self.encoder_blocks = nn.ModuleList([
            STEncoderBlock(
                dim=self.embed_dim, s_attn_size=self.s_attn_size, t_attn_size=self.t_attn_size,
                geo_num_heads=geo_num_heads, sem_num_heads=sem_num_heads, t_num_heads=t_num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=enc_dpr[i],
                act_layer=nn.GELU,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), device=self.device, type_ln=type_ln,
                output_dim=self.output_dim, num_nodes=self.num_nodes
            ) for i in range(enc_depth)
        ])

        # 跳跃层
        self.skip_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.embed_dim, out_channels=self.skip_dim, kernel_size=1,
            ) for _ in range(enc_depth)
        ])

        self.end_conv1 = nn.Conv2d(
            in_channels=self.input_window, out_channels=self.output_window, kernel_size=1, bias=True,
        )
        self.end_conv2 = nn.Conv2d(
            in_channels=self.skip_dim, out_channels=self.output_dim, kernel_size=1, bias=True,
        )

        tempp = np.load("/home/liuzemu/DGSTA/libcity/cache/dataset_cache/" + self.dataset + "/tempp.npy")
        tempp = torch.from_numpy(tempp).to(torch.float32)
        tempp = norm_embedding(tempp)
        self.tempp = self.cal_lape_emb(tempp).to(self.device)


    def cal_lape_emb(self, adj):
        adj = sp.coo_matrix(adj)
        d = np.array(adj.sum(1))
        isolated_point_num = np.sum(np.where(d, 0, 1))
        self._logger.info(f"Number of isolated points: {isolated_point_num}")
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        L = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort()
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
        laplacian_pe = torch.from_numpy(EigVec[:, isolated_point_num + 1: self.lape_dim + isolated_point_num + 1]).float()
        laplacian_pe.require_grad = False
        return laplacian_pe

    def forward(self, batch, lap_mx=None):
        x = batch['X']
        ind = batch['ind']  # new~~~~~~~~~~~~~~~~~~~~~~~
        T = x.shape[1]
        x_pattern_list = []
        for i in range(self.s_attn_size):
            x_pattern = F.pad(
                x[:, :T + i + 1 - self.s_attn_size, :, :self.output_dim],
                (0, 0, 0, 0, self.s_attn_size - 1 - i, 0),
                "constant", 0,
            ).unsqueeze(-2)
            x_pattern_list.append(x_pattern)
        x_patterns = torch.cat(x_pattern_list, dim=-2)  # (16,12,170,3,1)

        x_pattern_list = []
        pattern_key_list = []
        for i in range(self.output_dim):
            x_pattern_list.append(self.pattern_embeddings[i](x_patterns[..., i]).unsqueeze(-1))  # (16,12,170,64,1)
            pattern_key_list.append(self.pattern_embeddings[i](self.pattern_keys[..., i]).unsqueeze(-1))  # (16,64,1)
        x_patterns = torch.cat(x_pattern_list, dim=-1)  # (16,12,170,64,1)
        pattern_keys = torch.cat(pattern_key_list, dim=-1)  # (16,64,1)

        tempp = self.tempp
        enc = self.enc_embed_layer(x, lap_mx, tempp)
        skip = 0
        for i, encoder_block in enumerate(self.encoder_blocks):
            enc = encoder_block(enc, ind, x_patterns, pattern_keys, self.geo_mask, self.sem_mask)
            skip += self.skip_convs[i](enc.permute(0, 3, 2, 1))

        skip = self.end_conv1(F.relu(skip.permute(0, 3, 2, 1)))
        skip = self.end_conv2(F.relu(skip.permute(0, 3, 2, 1)))
        return skip.permute(0, 3, 2, 1)

    # 选择损失函数
    def get_loss_func(self, set_loss):
        if set_loss.lower() not in ['mae', 'mse', 'rmse', 'mape', 'logcosh', 'huber', 'quantile', 'masked_mae',
                                    'masked_mse', 'masked_rmse', 'masked_mape', 'masked_huber', 'r2', 'evar']:
            self._logger.warning('Received unrecognized train loss function, set default mae loss func.')
        if set_loss.lower() == 'mae':
            lf = loss.masked_mae_torch
        elif set_loss.lower() == 'mse':
            lf = loss.masked_mse_torch
        elif set_loss.lower() == 'rmse':
            lf = loss.masked_rmse_torch
        elif set_loss.lower() == 'mape':
            lf = loss.masked_mape_torch
        elif set_loss.lower() == 'logcosh':
            lf = loss.log_cosh_loss
        elif set_loss.lower() == 'huber':  # true
            lf = partial(loss.huber_loss, delta=self.huber_delta)
            # 返回一个函数，默认delta=2
        elif set_loss.lower() == 'quantile':
            lf = partial(loss.quantile_loss, delta=self.quan_delta)
        elif set_loss.lower() == 'masked_mae':
            lf = partial(loss.masked_mae_torch, null_val=0)
        elif set_loss.lower() == 'masked_mse':
            lf = partial(loss.masked_mse_torch, null_val=0)
        elif set_loss.lower() == 'masked_rmse':
            lf = partial(loss.masked_rmse_torch, null_val=0)
        elif set_loss.lower() == 'masked_mape':
            lf = partial(loss.masked_mape_torch, null_val=0)
        elif set_loss.lower() == 'masked_huber':
            lf = partial(loss.masked_huber_loss, delta=self.huber_delta, null_val=0)
        elif set_loss.lower() == 'r2':
            lf = loss.r2_score_torch
        elif set_loss.lower() == 'evar':
            lf = loss.explained_variance_score_torch
        else:
            lf = loss.masked_mae_torch
        return lf

    # 计算损失值
    def calculate_loss_without_predict(self, y_true, y_predicted, batches_seen=None, set_loss='masked_mae'):
        lf = self.get_loss_func(set_loss=set_loss)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])  # 标准化
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        if self.training:
            if batches_seen % self.step_size == 0 and self.task_level < self.output_window:
                self.task_level += 1
                self._logger.info('Training: task_level increase from {} to {}'.format(
                    self.task_level - 1, self.task_level))
                # self._logger.info('Current batches_seen is {}'.format(batches_seen))
            if self.use_curriculum_learning:  # true
                return lf(y_predicted[:, :self.task_level, :, :], y_true[:, :self.task_level, :, :])
            else:
                return lf(y_predicted, y_true)
        else:
            return lf(y_predicted, y_true)

    def calculate_loss(self, batch, batches_seen=None, lap_mx=None):
        y_true = batch['y']
        y_predicted = self.predict(batch, lap_mx)
        return self.calculate_loss_without_predict(y_true, y_predicted, batches_seen)

    def predict(self, batch, lap_mx=None):
        return self.forward(batch, lap_mx)
