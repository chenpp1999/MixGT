import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import copy
import math
# from tsformer import TSFormer
# from tsformerlib.transformer_layers import TransformerLayers
import yaml

from model.tsformer import TSFormer
from model.tsformerlib.transformer_layers import TransformerLayers

device = 'cuda'


class FeaLayer:
    def __init__(self, **kwargs):
        self.embed_dim = int(kwargs.get('embed_dim'))
        self.decoder_depth = int(kwargs.get('decoder_depth'))
        self.mlp_ratio = int(kwargs.get('mlp_ratio'))
        self.dropout = float(kwargs.get('dropout'))
        self.num_heads = int(kwargs.get('num_heads'))
        self.num_nodes = int(kwargs.get('num_nodes'))


# Correlation coefficient relationship
class AsLayer(nn.Module, FeaLayer):
    """鐢ㄤ簬鐢熸垚鐩稿叧鎬х煩闃碉紝X*XT
    It is used to generate the correlation matrix, X*XT"""

    def __init__(self, **kwargs):
        super(AsLayer, self).__init__()
        FeaLayer.__init__(self, **kwargs)
        self.layer1 = TransformerLayers(self.embed_dim, self.decoder_depth, self.mlp_ratio, self.num_heads,
                                        self.dropout)

    #     self.W1 = nn.Parameter(torch.FloatTensor(self.num_nodes, self.num_nodes), requires_grad=True)
    #     self.b1 = nn.Parameter(torch.FloatTensor(self.num_nodes), requires_grad=True)
    #     self.reser_parameter()
    #
    # def reser_parameter(self):
    #     nn.init.xavier_normal_(self.W1)
    #     nn.init.constant_(self.b1, 0.1)

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
        # 璁＄畻鑼冩暟
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        dot_product = torch.matmul(x, x.transpose(-2, -1))
        # print('dot', dot_product)
        output = dot_product / (torch.matmul(x_norm, x_norm.transpose(-2, -1)) + 0.00000001)
        return output


class CsmLayer(nn.Module, FeaLayer):
    """鐢ㄤ簬鐢熸垚NCS鐭╅樀锛屽嵆鑺傜偣寮哄害鐭╅樀
    It is used to generate the NCS matrix, that is, the nodal intensity matrix
    impA = (flow/all_flow*W1 + feture1/all_feature1*W2 + feature2/all_feature2*W3)
    impAB = (impA + impB)/2"""

    def __init__(self, **kwargs):
        super(CsmLayer, self).__init__()
        FeaLayer.__init__(self, **kwargs)
        self.tsformer = TransformerLayers(self.embed_dim, self.decoder_depth, self.mlp_ratio, self.num_heads,
                                          self.dropout)
        self.linear = nn.Linear(self.embed_dim, 2)
        self.dropout1 = torch.nn.Dropout(self.dropout)
        self.mlp = torch.nn.Conv1d(3, 1, kernel_size=1, padding=0, stride=1, bias=True)

    # def forward(self, inputs, org):
    #     x = self.tsformer(inputs)
    #     x = self.linear(self.dropout1(x))
    #     # print("ts_after", x.shape)
    #     org = org.transpose(1, 0).unsqueeze(-1)
    #     x = torch.cat([x, org], dim=-1)
    #     x = torch.softmax(x, dim=-2)
    #     x = x.transpose(-1, -2)
    #     # print(x.shape)
    #     x = x.squeeze(1) * 10
    #     # print(x)
    #     x = self.mlp(x)
    #     # print(x)
    #     # x = torch.matmul(x, self.W123)
    #     # x = torch.sum(x, dim=-2)
    #     # print(x.shape)
    #     x = x.unsqueeze(-1)
    #     # ncs_a = (x + x.transpose(-1, -2))/2
    #     ncs_a = (x + x.transpose(-1, -2)).squeeze(1)
    #     # print(ncs_a)
    #     # print('鐗瑰緛鐭╅樀', x.shape)
    #     return ncs_a

    def forward(self, inputs, org):
        x = self.tsformer(inputs)
        x = self.linear(self.dropout1(x))
        # print("ts_after", x.shape)
        org = org.transpose(1, 0).unsqueeze(-1)
        x = torch.cat([x, org], dim=-1)
        max1 = x.max(dim=2, keepdim=True).values
        min1 = x.min(dim=2, keepdim=True).values
        x = (x - min1) / (max1 - min1)
        x = x.transpose(-1, -2).squeeze(1)
        # x = x.squeeze(1) * 10
        x = self.mlp(x)
        max2 = x.max(dim=2, keepdim=True).values
        min2 = x.min(dim=2, keepdim=True).values
        x = (x - min2) / (max2 - min2)
        x = x.unsqueeze(-1)/2.0
        ncs_a = (x + x.transpose(-1, -2)).squeeze(1)
        return ncs_a


class CpModel(nn.Module):
    def __init__(self, **kwargs):
        super(CpModel, self).__init__()
        self.tsformer = TSFormer(**kwargs)
        self.pre_trained_tsformer_path = kwargs.get("pre_dir")
        self.aslayer = AsLayer(**kwargs)
        self.csmlayer = CsmLayer(**kwargs)
        self.load_pre_trained_model()

    def load_pre_trained_model(self):
        """Load pre-trained model"""
        # load parameters
        checkpoint_dict = torch.load(self.pre_trained_tsformer_path)
        self.tsformer.load_state_dict(checkpoint_dict["model_state_dict"])
        # freeze parameters
        for param in self.tsformer.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        '''
        :param inputs:[B, T, N]
        :return: [B, N, N]
        '''
        # print(inputs.shape)
        x0 = inputs.transpose(0, 1)
        x0 = x0[-1:]
        x = self.tsformer(x0)
        # print("xxxxxxxxx", x.shape)
        # batchsize, s_len, num_nodes, hiddens = x.shape
        # 鍥�1 鍏崇郴鐩镐技鍥� Correlation coefficient relationship, 鍥�2 寮哄害鐭╅樀
        A_s, A_ncs = self.aslayer(x), self.csmlayer(x, x0)

        return A_s, A_ncs


def clones(module, N):
    '''
    Produce N identical layers.
    :param module: nn.Module
    :param N: int
    :return: torch.nn.ModuleList
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def calculate_laplacian_with_self_loop(matrix, device):
    with torch.no_grad():
        matrix = matrix + torch.eye(matrix.size(0)).to(device)
        row_sum = matrix.sum(1)
        d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        normalized_laplacian = (
            matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
        )
    return normalized_laplacian


def a_calculate_laplacian_with_self_loop(matrix):
    with torch.no_grad():
        matrix[matrix >= matrix.mean()] = 1
        matrix[matrix < matrix.mean()] = 0
        bs, n, n = matrix.shape
        row_sum = matrix.sum(2)
        d_inv_sqrt = torch.pow(row_sum, -0.5).reshape(bs, n, 1)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_eye = torch.eye(n, device=device)
        d_mat_inv_sqrt = torch.mul(d_inv_sqrt, d_eye)
        matrix1 = torch.matmul(matrix, d_mat_inv_sqrt).transpose(1, 2)
        normalized_laplacian = torch.matmul(matrix1, d_mat_inv_sqrt)
    return normalized_laplacian


class Dgcn(nn.Module):
    def __init__(self, adj, _max_diffusion_step, shape, length, device):
        super(Dgcn, self).__init__()
        self._max_diffusion_step = _max_diffusion_step
        self.support = calculate_laplacian_with_self_loop(torch.FloatTensor(adj).to(device), device)
        self.weights = torch.nn.Parameter(torch.empty(*shape, device=device))
        self.biases = torch.nn.Parameter(torch.empty(length, device=device))
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.weights)
        torch.nn.init.constant_(self.biases, 0.0)

    def forward(self, inputs):
        '''
        :param x: src: (batch_size, N, T*D)
        :return: (batch_size, N, T*D)
        '''
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size, num_nodes, input_size = inputs.shape
        x = inputs
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        x1 = torch.sparse.mm(self.support, x0)
        x = self._concat(x, x1)

        for k in range(2, self._max_diffusion_step + 1):
            x2 = 2 * torch.sparse.mm(self.support, x1) - x0
            x = self._concat(x, x2)
            x1, x0 = x2, x1

        # num_matrices = len(self.support) * self._max_diffusion_step + 1  # Adds for x itself.
        num_matrices = self._max_diffusion_step + 1  # Adds for x itself.

        # [batchsize, numnodes, step*dim] torch.Size([64, 170, 432])
        x = torch.reshape(x, shape=[num_matrices, num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * num_nodes, input_size * num_matrices])
        x = torch.matmul(x, self.weights)  # (batch_size * self._num_nodes, output_size)
        x = x + self.biases
        x = x.reshape([batch_size, num_nodes, -1])
        # x = torch.sigmoid(x) # 鍔犱簡涓縺娲诲嚱鏁帮紝鐪嬫儏鍐垫敼鍙�
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return x

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)


class GCN(nn.Module):
    def __init__(self, seq_len_iput_dim: int, output_dim: int, **kwargs):
        super(GCN, self).__init__()

        self.seq_len_iput_dim = seq_len_iput_dim  # seq_len for prediction
        self._output_dim = output_dim  # hidden_dim for prediction
        self.weights = nn.Parameter(
            torch.FloatTensor(self.seq_len_iput_dim, self._output_dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain("tanh"))

    def forward(self, inputs, adj):
        '''
        :param x: src: (batch_size, N, T*D)
        :return: (batch_size, N, T*D)
        '''
        laplacian = a_calculate_laplacian_with_self_loop(adj)
        ax = torch.matmul(laplacian, inputs)
        outputs = ax
        return outputs


class InceptionCell(nn.Module):
    def __init__(self, seq_len, inputs_dim, outputs_dim, adj, shape, length, device,
                 _max_diffusion_step=2, spare_dim=1, fusio_dim=3, drop=0.2):
        super(InceptionCell, self).__init__()
        self.seq_len_inputsdim = int(seq_len * inputs_dim)
        self.gcn_adj1 = GCN(self.seq_len_inputsdim, self.seq_len_inputsdim)
        self.gcn_adj2 = GCN(self.seq_len_inputsdim, self.seq_len_inputsdim)
        self.dgcn = Dgcn(adj, _max_diffusion_step, shape, length, device)
        self.conv_spare = torch.nn.Conv2d(in_channels=spare_dim, out_channels=fusio_dim, kernel_size=3, padding=1,
                                          bias=False)
        self.conv_fuse = torch.nn.Conv2d(in_channels=fusio_dim, out_channels=spare_dim, kernel_size=3, padding=1,
                                         bias=False)
        self.attn_drop = nn.Dropout(drop)
        self.linear1 = nn.Linear(inputs_dim, inputs_dim, bias=True)
        '''
        璇ラ儴鍒嗙殑鍗风Н鍒嗙涓庡悎骞剁殑閫氶亾鍙€冭檻缁欏畠鎹㈡垚鍏朵粬浣嶇疆璇曡瘯锛屾瘮濡傛渶鍚庝竴缁�
        The volume integration and merging channels of this part can be considered for another position, 
        such as the last dimension
        '''

    def forward(self, inputs, adj1, adj2):
        '''
        :param x: src: (batch_size, T, N, D)
        :return: (batch_size, T, N, D)
        '''
        B, T, N, D = inputs.shape

        xi = inputs
        xii = xi.transpose(1, 2).reshape(B, N, T * D)
        xi = self.linear1(xi)
        xi = xi.transpose(1, 2).reshape(B, N, T * D).unsqueeze(1)
        xi = self.conv_spare(xi)
        x1 = xi[:, 0, :, :]
        x2 = xi[:, 1, :, :]
        x3 = xi[:, 2, :, :]

        x_adj1 = F.leaky_relu(self.gcn_adj1(x1, adj1).unsqueeze(1))
        x_adj2 = F.leaky_relu(self.gcn_adj2(x2, adj2).unsqueeze(1))
        x_dadj = F.leaky_relu(self.dgcn(x3).unsqueeze(1))
        hx = torch.cat([x_adj1, x_adj2, x_dadj], dim=1)

        xi = xii + self.conv_fuse(hx).squeeze(1)
        # xi = xii
        xi = self.att_fun(xi, xi, xi)
        outputs = xi.reshape(B, N, T, D).transpose(1, 2)
        return outputs

    def att_fun(self, q, k, v):
        d_k = q.size(-1)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        attn = attn.softmax(dim=-1)
        if self.attn_drop is not None:
            attn = self.attn_drop(attn)
        xs = torch.matmul(attn, v)
        return xs


class EncoderLayer(nn.Module):
    def __init__(self, seq_len, dim, outputs_dim, adj, shape, length, device, _max_diffusion_step=2,
                 spare_dim=1, fusio_dim=3, drop=0.2, norm_layer=None):
        super(EncoderLayer, self).__init__()
        if norm_layer:
            norm_layer = nn.LayerNorm
        self.incell = InceptionCell(seq_len, dim, outputs_dim, adj, shape, length, device,
                                    _max_diffusion_step, spare_dim, fusio_dim, drop)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = torch.nn.Linear(dim, dim)

    def forward(self, inputs, adj1, adj2):
        '''
        :param x: src: (batch_size, T, N, D)
        :return: (batch_size, T, N, D)
        '''
        # B, T, N, D = inputs.shape
        xe = inputs + self.norm1(self.incell(inputs, adj1, adj2))
        outputs = xe + self.norm2(self.mlp(xe))
        return outputs


class Encoder(nn.Module):
    def __init__(self, seq_len, N, dim, outputs_dim, adj, shape, length, device, _max_diffusion_step,
                 spare_dim, fusio_dim, drop, norm_layer):
        '''
        :param layer:  EncoderLayer
        :param N:  int, number of EncoderLayers
        '''
        super(Encoder, self).__init__()
        self.mlp_dim = nn.Linear(1, dim)
        self.layers = clones(EncoderLayer(seq_len, dim, outputs_dim, adj, shape, length, device,
                                          _max_diffusion_step, spare_dim, fusio_dim, drop,
                                          norm_layer), N)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, adj1, adj2):
        '''
        :param x: src: (batch_size, T, N)
        :return: (batch_size, T, N, D)
        '''
        # B, T, N = x.shape
        x = x.unsqueeze(-1)
        #         x = torch.relu(self.mlp_dim(x))
        x = F.leaky_relu(self.mlp_dim(x))
        for layer in self.layers:
            x = layer(x, adj1, adj2)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, nlayers, hidden_dim, num_heads, mlp_ratio, dropout):
        super(Decoder, self).__init__()
        self.d_model = hidden_dim
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * mlp_ratio, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, inputs):
        B, T, N, D = inputs.shape
        x = inputs * math.sqrt(self.d_model)
        x = x.transpose(1, 2)
        x = x.reshape(B * N, T, D).transpose(0, 1)
        outputs = self.transformer_encoder(x)
        outputs = outputs.reshape(B, N, T, D).transpose(1, 2)
        return outputs

import pandas as pd

class InceptionTransformer(nn.Module):
    def __init__(self, adj, kawrgs, pre_kwargs):
        super(InceptionTransformer, self).__init__()
        self.seq_len = int(kawrgs.get("seq_len"))
        self.pre_len = int(kawrgs.get("pre_len"))
        self.N = int(kawrgs.get("N"))  # 灞傛暟
        self.dim = int(kawrgs.get("dim"))
        self.outputs_dim = int(kawrgs.get("outputs_dim"))
        self.device = kawrgs.get("device")
        self._max_diffusion_step = int(kawrgs.get("_max_diffusion_step"))
        self.spare_dim = int(kawrgs.get("spare_dim"))
        self.fusio_dim = int(kawrgs.get("fusio_dim"))
        self.drop = float(kawrgs.get("drop"))
        self.norm_layer = kawrgs.get("norm_layer")
        self.head_nums = int(kawrgs.get("head_nums"))
        self.mlp_radio = int(kawrgs.get("mlp_radio"))
        self.drop_radio = float(kawrgs.get("drop_radio"))
        self.shape = [int(self.seq_len * self.dim * self.fusio_dim), int(self.seq_len * self.dim)]
        self.length = int(self.seq_len * self.dim)

        self.pre_model = CpModel(**pre_kwargs)
        self.encoder = Encoder(self.seq_len, self.N, self.dim, self.outputs_dim, adj, self.shape, self.length,
                               self.device, self._max_diffusion_step, self.spare_dim, self.fusio_dim, self.drop,
                               self.norm_layer)
        self.decoder = Decoder(self.N, self.dim, self.head_nums, self.mlp_radio, self.drop_radio)
        self.mlp = nn.Sequential(nn.Linear(self.outputs_dim, int(self.outputs_dim / 2)),
                                 nn.Linear(int(self.outputs_dim / 2), 1))
        self.a = 0

    def forward(self, x):
        # print("1", x.shape)
        x = x.transpose(0, 1)
        adj1, adj2 = self.pre_model(x)
        a1 = pd.DataFrame(adj1[-1, :, :].cpu().numpy())
        a2 = pd.DataFrame(torch.abs(adj2[30, :, :]).cpu().numpy())
        a1.to_csv('./at/{}adj1.csv'.format(self.a))
        a2.to_csv('./at/{}adj2.csv'.format(self.a))
        self.a+=1
        # 宸釜缂栫爜灞�
        # torch.autograd.set_detect_anomaly(True)
        #         x = self.dropout(x + self.pe)
        encoder_output = self.encoder(x, adj1, adj2)
        # output = self.decoder(x)
        # output = self.decoder(encoder_output)
        output = self.mlp(encoder_output).squeeze(-1)
        output = output.transpose(0, 1)
        return output

    def Loss_l2(self):
        base_params = dict(self.named_parameters())
        loss_l2 = 0
        count = 0
        for key, value in base_params.items():
            if 'bias' not in key:
                loss_l2 = loss_l2 + torch.sum(value ** 2)
                count = count + value.nelement()
        return loss_l2


if __name__ == "__main__":
    with open("C:\\Users\\15315\\Desktop\\conda_cp\\LPNet\\data\\model\\PEMS08.yaml") as f:
        config = yaml.safe_load(f)
        print(config)
        model_kwargs = config.get("model")
        print(model_kwargs)
        pre_model_kwargs = config.get("pre_model")
        x = torch.rand([64, 12, 170])
        adj = torch.rand([170, 170])
        model = InceptionTransformer(adj, model_kwargs, pre_model_kwargs)
        print(model)
        print(model(x).shape)
