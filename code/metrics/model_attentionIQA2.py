# flake8: noqa
import timm
import torch
from einops import rearrange
from timm.models.vision_transformer import Block
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class ChannelAttn(nn.Module):

    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(drop)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x


class SaveOutput:

    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


class AttentionIQA(nn.Module):

    def __init__(self,
                 embed_dim=72,
                 num_outputs=1,
                 patch_size=8,
                 drop=0.1,
                 depths=[2, 2],
                 window_size=4,
                 dim_mlp=768,
                 num_heads=[4, 4],
                 img_size=224,
                 num_channel_attn=2,
                 **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_size = img_size // patch_size
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)

        self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

        self.channel_attn1 = nn.Sequential(*[ChannelAttn(self.input_size**2) for i in range(num_channel_attn)])
        self.channel_attn2 = nn.Sequential(*[ChannelAttn(self.input_size**2) for i in range(num_channel_attn)])

        self.conv1 = nn.Conv2d(embed_dim * 4, embed_dim, 1, 1, 0)
        self.swintransformer1 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim,
            window_size=window_size,
            dim_mlp=dim_mlp)
        self.swintransformer2 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim // 2,
            window_size=window_size,
            dim_mlp=dim_mlp)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)

        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs), nn.ReLU())
        self.fc_weight = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs), nn.Sigmoid())

    def extract_feature(self, save_output):
        x6 = save_output.outputs[6][:, 1:]
        x7 = save_output.outputs[7][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x9 = save_output.outputs[9][:, 1:]
        x = torch.cat((x6, x7, x8, x9), dim=2)
        return x

    def forward(self, x):
        _x = self.vit(x)
        x = self.extract_feature(self.save_output)
        self.save_output.outputs.clear()

        # stage 1
        x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)
        x = self.channel_attn1(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv1(x)
        x = self.swintransformer1(x)

        # stage2
        x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        x = self.channel_attn2(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv2(x)
        x = self.swintransformer2(x)

        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        f = self.fc_score(x)
        w = self.fc_weight(x)
        s = torch.sum(f * w, dim=1) / torch.sum(w, dim=1)
        return s


""" attention decoder mask """


def get_attn_decoder_mask(seq):
    subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = subsequent_mask.triu(diagonal=1)  # upper triangular part of a matrix(2-D)
    return subsequent_mask


""" attention pad mask """


def get_attn_pad_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(i_pad)
    pad_attn_mask = pad_attn_mask.unsqueeze(1).expand(batch_size, len_q, len_k)
    return pad_attn_mask


class DecoderWindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.W_Q = nn.Linear(dim, dim)
        self.W_K = nn.Linear(dim, dim)
        self.W_V = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, attn_mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = q.shape

        q = self.W_Q(q).view(B_, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.W_K(k).view(B_, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.W_V(v).view(B_, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn.masked_fill_(attn_mask, -1e9)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


""" decoder layer """


class DecoderLayer(nn.Module):

    def __init__(self,
                 input_resolution=(28, 28),
                 embed_dim=256,
                 layer_norm_epsilon=1e-12,
                 dim_mlp=1024,
                 num_heads=4,
                 dim_head=128,
                 window_size=7,
                 shift_size=0,
                 i_layer=0,
                 act_layer=nn.GELU,
                 drop=0.,
                 drop_path=0.):
        super().__init__()
        self.i_layer = i_layer
        self.shift_size = shift_size
        self.window_size = window_size
        self.input_resolution = input_resolution
        self.conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_epsilon)
        self.dec_enc_attn_wmsa = DecoderWindowAttention(
            dim=embed_dim,
            window_size=to_2tuple(window_size),
            num_heads=num_heads,
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_epsilon)
        self.dec_enc_attn_swmsa = DecoderWindowAttention(
            dim=embed_dim,
            window_size=to_2tuple(window_size),
            num_heads=num_heads,
        )
        self.layer_norm3 = nn.LayerNorm(embed_dim, eps=layer_norm_epsilon)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=dim_mlp, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # calculate attention mask for SW-MSA
        H, W = self.input_resolution
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        self.register_buffer("attn_mask", attn_mask)

    def partition(self, inputs, B, H, W, C, shift_size=0):
        # partition mask_dec_inputs
        inputs = inputs.view(B, H, W, C)
        if shift_size > 0:
            shifted_inputs = torch.roll(inputs, shifts=(-shift_size, -shift_size), dims=(1, 2))
        else:
            shifted_inputs = inputs
        windows_inputs = window_partition(shifted_inputs, self.window_size)  # nW*B, window_size, window_size, C
        windows_inputs = windows_inputs.view(-1, self.window_size * self.window_size,
                                             C)  # nW*B, window_size*window_size, C
        return windows_inputs

    def reverse(self, inputs, B, H, W, C, shift_size=0):
        # merge windows
        inputs = inputs.view(-1, self.window_size, self.window_size, C)
        inputs = window_reverse(inputs, self.window_size, H, W)  # B H' W' C
        # reverse cyclic shift
        if shift_size > 0:
            inputs = torch.roll(inputs, shifts=(shift_size, shift_size), dims=(1, 2))
        else:
            inputs = inputs
        inputs = inputs.view(B, H * W, C)
        return inputs

    def forward(self, mask_dec_inputs, enc_outputs, self_attn_mask, dec_enc_attn_mask):
        H, W = self.input_resolution[0], self.input_resolution[1]
        B, L, C = mask_dec_inputs.shape
        assert L == H * W, "input feature has wrong size"

        dec_enc_att_outputs = mask_dec_inputs
        shortcut1 = dec_enc_att_outputs
        dec_enc_att_outputs = self.layer_norm1(dec_enc_att_outputs)
        enc_outputs = self.partition(enc_outputs, B, H, W, C, shift_size=0)
        dec_enc_att_outputs = self.partition(dec_enc_att_outputs, B, H, W, C, shift_size=0)
        dec_enc_att_outputs = self.dec_enc_attn_wmsa(
            q=dec_enc_att_outputs, k=enc_outputs, v=enc_outputs, mask=None, attn_mask=dec_enc_attn_mask)
        dec_enc_att_outputs = self.reverse(dec_enc_att_outputs, B, H, W, C, shift_size=0)
        enc_outputs = self.reverse(enc_outputs, B, H, W, C, shift_size=0)
        dec_enc_att_outputs = shortcut1 + self.drop_path(dec_enc_att_outputs)

        shortcut2 = dec_enc_att_outputs
        dec_enc_att_outputs = self.layer_norm2(dec_enc_att_outputs)
        enc_outputs = self.partition(enc_outputs, B, H, W, C, shift_size=self.window_size // 2)
        dec_enc_att_outputs = self.partition(dec_enc_att_outputs, B, H, W, C, shift_size=self.window_size // 2)
        dec_enc_att_outputs = self.dec_enc_attn_swmsa(
            q=dec_enc_att_outputs, k=enc_outputs, v=enc_outputs, mask=self.attn_mask, attn_mask=dec_enc_attn_mask)
        dec_enc_att_outputs = self.reverse(dec_enc_att_outputs, B, H, W, C, shift_size=self.window_size // 2)
        enc_outputs = self.reverse(enc_outputs, B, H, W, C, shift_size=self.window_size // 2)
        dec_enc_att_outputs = shortcut2 + self.drop_path(dec_enc_att_outputs)

        shortcut3 = dec_enc_att_outputs
        dec_enc_att_outputs = self.layer_norm3(dec_enc_att_outputs)
        dec_enc_att_outputs = self.mlp(dec_enc_att_outputs)
        dec_enc_att_outputs = shortcut3 + self.drop_path(dec_enc_att_outputs)

        dec_enc_att_outputs = rearrange(
            dec_enc_att_outputs, 'b (h w) c -> b c h w', h=self.input_resolution[0], w=self.input_resolution[1])
        # if self.i_layer % 2 == 0:
        #     dec_enc_att_outputs = self.conv(dec_enc_att_outputs)
        dec_enc_att_outputs = rearrange(dec_enc_att_outputs, 'b c h w -> b (h w) c')
        return dec_enc_att_outputs


""" decoder """


class SwinDecoder(nn.Module):

    def __init__(self,
                 input_resolution=(28, 28),
                 embed_dim=256,
                 num_heads=4,
                 num_layers=2,
                 drop=0.1,
                 i_pad=0,
                 dim_mlp=1024,
                 window_size=7,
                 drop_path_rate=0.1):
        super().__init__()
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.i_pad = i_pad
        self.dropout = nn.Dropout(drop)
        self.layers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        for i_layer in range(num_layers):
            layer = DecoderLayer(
                input_resolution=(input_resolution[0], input_resolution[1]),
                embed_dim=embed_dim,
                dim_mlp=dim_mlp,
                window_size=window_size,
                i_layer=i_layer + 1,
                shift_size=window_size // 2,
                drop_path=dpr[i_layer])
            self.layers.append(layer)

    def forward(self, y_embed, enc_outputs):
        inputs_embed = y_embed
        B, C, H, W = y_embed.shape
        inputs_embed = rearrange(inputs_embed, 'b c h w -> b (h w) c')
        dec_outputs = self.dropout(inputs_embed)

        idx = 1
        down_rate = 0
        for layer in self.layers:
            window_num = int((self.input_resolution[0] // 2 ** down_rate) // self.window_size) * \
                         int((self.input_resolution[1] // 2 ** down_rate) // self.window_size)
            enc_inputs_length = self.window_size * self.window_size
            dec_inputs_length = self.window_size * self.window_size
            mask_enc_inputs = torch.ones(B * window_num, enc_inputs_length).cuda()
            mask_dec_inputs = torch.ones(B * window_num, dec_inputs_length).cuda()

            dec_attn_pad_mask = get_attn_pad_mask(mask_dec_inputs, mask_dec_inputs, self.i_pad)
            dec_attn_decoder_mask = get_attn_decoder_mask(mask_dec_inputs)
            dec_self_attn_mask = torch.gt((dec_attn_pad_mask + dec_attn_decoder_mask), 0)
            dec_enc_attn_mask = get_attn_pad_mask(mask_dec_inputs, mask_enc_inputs, self.i_pad)

            dec_self_attn_mask = dec_self_attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            dec_enc_attn_mask = dec_enc_attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

            dec_outputs = layer(dec_outputs, enc_outputs[idx - 1], dec_self_attn_mask, dec_enc_attn_mask)
            idx += 1
        dec_outputs = rearrange(
            dec_outputs,
            'b (h w) c -> b c h w',
            h=self.input_resolution[0] // 2**down_rate,
            w=self.input_resolution[1] // 2**down_rate)
        return dec_outputs


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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 dim_mlp=1024.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.dim_mlp = dim_mlp
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = self.dim_mlp
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                           -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                           -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 dim_mlp=1024,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                dim_mlp=dim_mlp,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.input_resolution[0], w=self.input_resolution[1])
        x = F.relu(self.conv(x))
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class SwinTransformer(nn.Module):

    def __init__(self,
                 patches_resolution,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 embed_dim=256,
                 drop=0.1,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 dropout=0.,
                 window_size=7,
                 dim_mlp=1024,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.dropout = nn.Dropout(p=drop)
        self.num_features = embed_dim
        self.num_layers = len(depths)
        self.patches_resolution = (patches_resolution[0], patches_resolution[1])
        self.downsample = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, stride=2, padding=1)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=self.embed_dim,
                input_resolution=patches_resolution,
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                dim_mlp=dim_mlp,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=dropout,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=downsample,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

    def forward(self, x):
        x = self.dropout(x)
        x = self.pos_drop(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for layer in self.layers:
            _x = x
            x = layer(x)
            x = 0.13 * x + _x
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.patches_resolution[0], w=self.patches_resolution[1])
        return x
