import torch
from torch import nn
from timm.models.vision_transformer import trunc_normal_

from model.tsformerlib.patch import PatchEmbedding
from model.tsformerlib.mask import MaskGenerator
from model.tsformerlib.positional_encoding import PositionalEncoding
from model.tsformerlib.transformer_layers import TransformerLayers

# from tsformerlib.patch import PatchEmbedding
# from tsformerlib.mask import MaskGenerator
# from tsformerlib.positional_encoding import PositionalEncoding
# from tsformerlib.transformer_layers import TransformerLayers

device = 'cuda'

def unshuffle(shuffled_tokens):
    dic = {}
    for k, v, in enumerate(shuffled_tokens):
        dic[v] = k
    unshuffle_index = []
    for i in range(len(shuffled_tokens)):
        unshuffle_index.append(dic[i])
    return unshuffle_index

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TSFormer(nn.Module):
    """An efficient unsupervised pre-training model for Time Series based on transFormer blocks. (TSFormer)"""

    def __init__(self,**kwargs):
        super().__init__()
        assert kwargs.get('mode') in ["pre-train", "forecasting"], "Error mode."
        self.patch_size = int(kwargs.get('patch_size'))
        self.in_channel = int(kwargs.get('in_channel'))
        self.embed_dim = int(kwargs.get('embed_dim'))
        self.num_heads = int(kwargs.get('num_heads'))
        self.num_token = int(kwargs.get('num_token'))
        self.mask_ratio = float(kwargs.get('mask_ratio'))
        self.dropout = float(kwargs.get('dropout'))
        self.mode = kwargs.get('mode')
        self.decoder_depth = int(kwargs.get('decoder_depth'))
        self.encoder_depth = int(kwargs.get('encoder_depth'))
        self.mlp_ratio = int(kwargs.get('mlp_ratio'))
        self.selected_feature = 0

        # norm layers
        self.encoder_norm = nn.LayerNorm(self.embed_dim)
        self.decoder_norm = nn.LayerNorm(self.embed_dim)

        # encoder specifics
        # # patchify & embedding
        self.patch_embedding = PatchEmbedding(self.patch_size, self.in_channel, self.embed_dim, norm_layer=None)
        # # positional encoding
        self.positional_encoding = PositionalEncoding(self.embed_dim, dropout=self.dropout)
        # # masking
        self.mask = MaskGenerator(self.num_token, self.mask_ratio)
        # encoder
        self.encoder = TransformerLayers(self.embed_dim, self.encoder_depth, self.mlp_ratio, self.num_heads, self.dropout)

        # decoder specifics
        # transform layer
        self.enc_2_dec_emb = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        # # mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.embed_dim))
        # # decoder
        self.decoder = TransformerLayers(self.embed_dim, self.decoder_depth, self.mlp_ratio, self.num_heads, self.dropout)

        # # prediction (reconstruction) layer
        self.output_layer = nn.Linear(self.embed_dim, self.patch_size)
        self.initialize_weights()

    def initialize_weights(self):
        # positional encoding
        nn.init.uniform_(self.positional_encoding.position_embedding, -.02, .02)
        # mask token
        trunc_normal_(self.mask_token, std=.02)

    def encoding(self, long_term_history, mask=True):
        """Encoding process of TSFormer: patchify, positional encoding, mask, Transformer layers.

        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS with shape [B, N, 1, P * L],
                                                which is used in the TSFormer.
                                                P is the number of segments (patches).
            mask (bool): True in pre-training stage and False in forecasting stage.

        Returns:
            torch.Tensor: hidden states of unmasked tokens
            list: unmasked token index
            list: masked token index
        """

        batch_size, num_nodes, _, _ = long_term_history.shape
        # print("tsformer_long_term_history: ", long_term_history.shape)
        # patchify and embed input
        patches = self.patch_embedding(long_term_history)     # B, N, d, P
        # print('pathch_embedding: ', patches.shape)
        patches = patches.transpose(-1, -2)         # B, N, P, d
        # print(patches.shape)
        # positional embedding
        patches = self.positional_encoding(patches)
        # print("positional_encoding: ", patches.shape)
        # mask
        if mask:
            unmasked_token_index, masked_token_index = self.mask()
            # encoder_input = patches[:, :, unmasked_token_index, :]
            encoder_input = patches[:, unmasked_token_index, :, :]
        else:
            unmasked_token_index, masked_token_index = None, None
            encoder_input = patches
        # encoding
        # print("encoder", encoder_input.shape)
        encoder_input = encoder_input.transpose(1, 2)
        # print("encoder_trans", encoder_input.shape)
        hidden_states_unmasked = self.encoder(encoder_input)
        # hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1, self.embed_dim)
        hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, 1, -1, self.embed_dim)
        # print("tsformer_hidden_states: ", hidden_states_unmasked.shape)
        return hidden_states_unmasked, unmasked_token_index, masked_token_index

    def decoding(self, hidden_states_unmasked, masked_token_index):
        """Decoding process of TSFormer: encoder 2 decoder layer, add mask tokens, Transformer layers, predict.

        Args:
            hidden_states_unmasked (torch.Tensor): hidden states of masked tokens [B, N, P*(1-r), d].
            masked_token_index (list): masked token index

        Returns:
            torch.Tensor: reconstructed data
        """
        # print('-------decoding--------')
        batch_size, pre_len, _, _ = hidden_states_unmasked.shape
        # print('进：', hidden_states_unmasked.shape)
        # encoder 2 decoder layer
        hidden_states_unmasked = self.enc_2_dec_emb(hidden_states_unmasked)
        # print('enc_2_dec_emb: ', hidden_states_unmasked.shape)
        # add mask tokens
        hidden_states_masked = self.positional_encoding(
            self.mask_token.expand(batch_size, pre_len, len(masked_token_index), hidden_states_unmasked.shape[-1]),
            index=masked_token_index
            )
        # print('positonal_enconding', hidden_states_masked.shape)
        hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)   # B, N, P, d
        # print('cat: ', hidden_states_full.shape)
        # decoding
        hidden_states_full = self.decoder(hidden_states_full)
        hidden_states_full = self.decoder_norm(hidden_states_full)

        # prediction (reconstruction)
        reconstruction_full = self.output_layer(hidden_states_full.view(batch_size, pre_len, -1, self.embed_dim))
        # print('----------------------')
        return reconstruction_full

    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full, unmasked_token_index, masked_token_index):
        """Get reconstructed masked tokens and corresponding ground-truth for subsequent loss computing.

        Args:
            reconstruction_full (torch.Tensor): reconstructed full tokens.
            real_value_full (torch.Tensor): ground truth full tokens.
            unmasked_token_index (list): unmasked token index.
            masked_token_index (list): masked token index.

        Returns:
            torch.Tensor: reconstructed masked tokens.
            torch.Tensor: ground truth masked tokens.
        """
        # get reconstructed masked tokens
        batch_size, pre_len, numnodes, hiddens = reconstruction_full.shape
        reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_index):, :]     # B, N, r*P, d
        # print('recons: ', reconstruction_masked_tokens.shape)
        # reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, pre_len, -1).transpose(1, 2)     # B, r*P*d, N
        # print('view_recons: ', reconstruction_masked_tokens.shape)
        # label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)[:, :, :, self.selected_feature, :].transpose(1, 2)  # B, N, P, L
        label_full = real_value_full.permute(0, 2, 1, 3)
        # print('labe: ', label_full.shape)
        label_masked_tokens = label_full[:, :, masked_token_index, :].contiguous() # B, N, r*P, d
        # label_masked_tokens = label_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)  # B, r*P*d, N

        return reconstruction_masked_tokens, label_masked_tokens

    # def Loss_l2(self):
    #     base_params = dict(self.named_parameters())
    #     loss_l2 = 0
    #     count = 0
    #     for key, value in base_params.items():
    #         if 'bias' not in key:
    #             loss_l2 += torch.sum(value**2)
    #             count += value.nelement()
    #     return loss_l2

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, epoch: int = None, **kwargs) -> torch.Tensor:
        """feed forward of the TSFormer.
            TSFormer has two modes: the pre-training mode and the forecasting mode,
                                    which are used in the pre-training stage and the forecasting stage, respectively.

        Args:
            history_data (torch.Tensor): very long-term historical time series with shape B, L * P, N, 1.

        Returns:
            pre-training:
                torch.Tensor: the reconstruction of the masked tokens. Shape [B, L * P * r, N, 1]
                torch.Tensor: the ground truth of the masked tokens. Shape [B, L * P * r, N, 1]
                dict: data for plotting.
            forecasting:
                torch.Tensor: the output of TSFormer of the encoder with shape [B, N, L, 1].
        """
        # reshape
        # print(history_data.shape)
        history_data = history_data.unsqueeze(-1)
        history_data = history_data.permute(1, 3, 2, 0)
        history_data = history_data.permute(0, 2, 3, 1)     # B, N, 1, L * P
        # print('his: ', history_data.shape)
        # feed forward
        if self.mode == "pre-train":
            # encoding
            hidden_states_unmasked, unmasked_token_index, masked_token_index = self.encoding(history_data)
            print(hidden_states_unmasked.shape)
            # print(hidden_states_unmasked.shape)
            # decoding
            reconstruction_full = self.decoding(hidden_states_unmasked, masked_token_index)
            # for subsequent loss computing
            # print(reconstruction_full.shape)
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(reconstruction_full, history_data, unmasked_token_index, masked_token_index)
            return reconstruction_masked_tokens, label_masked_tokens, masked_token_index
        else:
            hidden_states_full, _, _ = self.encoding(history_data, mask=False)
            return hidden_states_full

# num_token=288 * 2 * 7/12，设定为两周的长度288*2*7，除以12，即以12个分一组，在数据中的体现是将卷积的卷积核大小设置为（12， 1），卷积的步长设置为12
if __name__=='__main__':
    import matplotlib.pyplot as plt
    # pre-train
    model = TSFormer(logger=None, patch_size=1, in_channel=1, embed_dim=64, num_heads=4, mlp_ratio=4, dropout=0.1, num_token=170,
                     mask_ratio=0.6, encoder_depth=4, decoder_depth=1, mode='pre-train').to(device)
    historybn = torch.rand([1, 64, 170]).to(device)
    # forecasting
    # model = TSFormer(patch_size=12, in_channel=1, embed_dim=96, num_heads=4, mlp_ratio=4, dropout=0.1, num_token=288 * 2 * 7/12,
    #                  mask_ratio=0.75, encoder_depth=4, decoder_depth=1, mode='forecasting').to(device)
    # historybn = torch.rand([256, 36, 170, 1]).to(device)

    # checkpoint_dict = torch.load('C:\\Users\\15315\\Desktop\\conda_cp\\STEP\\tsformer_ckpt\\TSFormer_PEMS04.pt')
    # model.load_state_dict(checkpoint_dict["model_state_dict"])

    pre, y, _ = model(historybn)
    print(y.shape)
    print(pre.shape)
    x = range(len(y[0, 0, :, 0]))
    plt.plot(x, pre[0, 0, :, 0].detach().cpu().numpy(), 'r-', alpha=1, color='gold', linewidth=1, label='pre')
    plt.plot(x, y[0, 0, :, 0].detach().cpu().numpy(), 'r-', alpha=0.5, color='green', linewidth=1, label='y')
    plt.legend(loc="upper right")
    plt.xlabel('time_steps')
    plt.ylabel('flow')
    plt.show()
