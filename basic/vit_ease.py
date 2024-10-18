import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from timm.models.vision_transformer import PatchEmbed
import timm
from functools import partial
from collections import OrderedDict
import copy


class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        # Adapter layer normalization
        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd) if adapter_layernorm_option in ["in", "out"] else None

        # Adapter scaling (either learnable or fixed scalar)
        self.scale = nn.Parameter(torch.ones(1)) if adapter_scalar == "learnable_scalar" else float(adapter_scalar)

        # Down-projection, non-linearity, and up-projection
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        # Optional dropout and initialization
        self.dropout = dropout
        if init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        return up + residual if add_residual else up


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        B, N, C = x.shape

        q = self._shape(self.q_proj(x), N, B).view(B * self.num_heads, -1, self.head_dim)
        k = self._shape(self.k_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(self.v_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(attn_output))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None):
        super().__init__()
        self.config = config
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

    def forward(self, x, adapt=None):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if adapt is not None:
            adapt_x = adapt(x, add_residual=False)
        else:
            adapt_x = None

        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))

        if adapt_x is not None:
            if self.config.ffn_adapt:
                if self.config.ffn_option == 'sequential':
                    x = adapt(x)
                elif self.config.ffn_option == 'parallel':
                    x = x + adapt_x

        return residual + x


class VisionTransformer(nn.Module):
    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, 
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None, 
                 act_layer=None, tuning_config=None):
        super().__init__()
        self.tuning_config = tuning_config
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, 
                  attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, 
                  config=tuning_config, layer_id=i)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([('fc', nn.Linear(embed_dim, representation_size)), ('act', nn.Tanh())]))
        else:
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if distilled and num_classes > 0 else None

        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

        if tuning_config.vpt_on:
            assert tuning_config.vpt_num > 0
            self.embeddings = nn.ParameterList([nn.Parameter(torch.empty(1, tuning_config.vpt_num, embed_dim)) for _ in range(depth)])
            for eee in self.embeddings:
                torch.nn.init.xavier_uniform_(eee.data)

        self.adapter_list = []
        self.cur_adapter = nn.ModuleList()
        self.get_new_adapter()

    def get_new_adapter(self):
        config = self.tuning_config
        self.cur_adapter = nn.ModuleList()
        if config.ffn_adapt:
            for i in range(len(self.blocks)):
                adapter = Adapter(config=config, dropout=0.1, bottleneck=config.ffn_num, 
                                  init_option=config.ffn_adapter_init_option, 
                                  adapter_scalar=config.ffn_adapter_scalar, 
                                  adapter_layernorm_option=config.ffn_adapter_layernorm_option).to(self.tuning_config._device)
                self.cur_adapter.append(adapter)
            self.cur_adapter.requires_grad_(True)

    def forward_train(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for i, blk in enumerate(self.blocks):
            if self.tuning_config.vpt_on:
                prompt_tokens = self.embeddings[i].expand(B, -1, -1)
                x = torch.cat([prompt_tokens, x], dim=1)
            x = blk(x, self.cur_adapter[i])

        return self.norm(x[:, 0]) if not self.global_pool else self.fc_norm(x[:, 1:].mean(1))

    def forward_test(self, x):
        return self.forward_train(x)

    def forward(self, x, test=False):
        return self.forward_test(x) if test else self.forward_train(x)


def vit_base_patch16_224_ease(pretrained=False, **kwargs):
    # Initialize Vision Transformer model
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    # Load a pre-trained ViT model with no classification head (num_classes=0)
    checkpoint_model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()

    # Adjust the state dictionary to match the VisionTransformer model
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            # Split qkv weights into separate q, k, v weights
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = qkv_weight[:768]
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = qkv_weight[768:768 * 2]
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = qkv_weight[768 * 2:]
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            # Split qkv biases into separate q, k, v biases
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = qkv_bias[:768]
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = qkv_bias[768:768 * 2]
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = qkv_bias[768 * 2:]

        # Rename mlp.fc.* to match the model's fc layers
        elif 'mlp.fc' in key:
            state_dict[key.replace('mlp.', '')] = state_dict.pop(key)

    # Load the state dict into the VisionTransformer model
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # Freeze all parameters except for the adapter layers
    for name, p in model.named_parameters():
        p.requires_grad = name in msg.missing_keys

    return model


def vit_base_patch16_224_in21k_ease(pretrained=False, **kwargs):
    # Initialize Vision Transformer model
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    # Load a pre-trained ViT model with no classification head (num_classes=0)
    checkpoint_model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()

    # Adjust the state dictionary to match the VisionTransformer model
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            # Split qkv weights into separate q, k, v weights
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = qkv_weight[:768]
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = qkv_weight[768:768 * 2]
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = qkv_weight[768 * 2:]
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            # Split qkv biases into separate q, k, v biases
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = qkv_bias[:768]
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = qkv_bias[768:768 * 2]
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = qkv_bias[768 * 2:]

        # Rename mlp.fc.* to match the model's fc layers
        elif 'mlp.fc' in key:
            state_dict[key.replace('mlp.', '')] = state_dict.pop(key)

    # Load the state dict into the VisionTransformer model
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # Freeze all parameters except for the adapter layers
    for name, p in model.named_parameters():
        p.requires_grad = name in msg.missing_keys

    return model






