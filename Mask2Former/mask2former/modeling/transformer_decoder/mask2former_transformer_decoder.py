# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
from copy import deepcopy
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch, time, os
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.nonlin(out)
        return out

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.plane_para_embed = nn.Linear(hidden_dim, 3)
            # self.plane_para_embed = nn.Linear(hidden_dim, 4)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)


        # zmf
        self.rgb_embed = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=mask_dim*4,num_layers=3)
        self.convs = nn.ModuleDict()
        self.use_feat_mask = False

        self.convs['0_1'] = Conv3x3(768, 256)
        self.convs['1_1'] = Conv3x3(512, 256)
        self.convs['2_1'] = Conv3x3(320, 256)
        self.convs['-1_1'] = ConvBlock(320, 256)
        self.convs['-1_2'] = Conv3x3(256, 256)
        
        if self.use_feat_mask:
            self.convs['paint'] = Conv3x3(num_queries*3, num_queries*3)

        # self.convs['m2a'] = Conv3x3(num_queries, num_queries)
        self.sigmoid = nn.Sigmoid()
        # fmz

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        return ret

    def forward(self, x, mask_features, mask = None, nonplane_rgba_list = None, encoder_feat = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = [] # [H/32,W/32],[H/16,W/16],[H/8,W/8]

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels): # 3
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_plane_para = []
        predictions_mask = []
        plane_rgba_list = []


        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask, outputs_plane_para, outputs_rgba = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0],encoder_feat=encoder_feat, ii=-1)

        predictions_class.append(outputs_class)
        predictions_plane_para.append(outputs_plane_para)
        predictions_mask.append(outputs_mask)
        plane_rgba_list.append(outputs_rgba)

        for i in range(self.num_layers): # 9
            level_index = i % self.num_feature_levels # 3
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask, outputs_plane_para, outputs_rgba = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],encoder_feat=encoder_feat,ii=i)
            predictions_class.append(outputs_class)
            predictions_plane_para.append(outputs_plane_para)
            predictions_mask.append(outputs_mask)
            if i in [0,1,2]:
                plane_rgba_list.append(outputs_rgba)

        assert len(predictions_class) == self.num_layers + 1
        assert len(predictions_plane_para) == self.num_layers + 1

        plane_rgba_list = [plane_rgba_list[0], plane_rgba_list[3], plane_rgba_list[2], plane_rgba_list[1]]

        out = {
            'pred_logits': predictions_class[-1],
            'pred_plane_para': predictions_plane_para[-1],
            'pred_masks': predictions_mask[-1],
            'pred_nonplane_rgba': nonplane_rgba_list,
            'pred_plane_rgba': plane_rgba_list,
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask, predictions_plane_para
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features_B256hw, attn_mask_target_size, encoder_feat, ii):
        tf_decoder_output_BQ256 = self.decoder_norm(output) # Q B 256
        tf_decoder_output_BQ256 = tf_decoder_output_BQ256.transpose(0, 1)

        outputs_class = self.class_embed(tf_decoder_output_BQ256)
        outputs_plane_para = self.plane_para_embed(tf_decoder_output_BQ256)
        mask_embed_BQ256 = self.mask_embed(tf_decoder_output_BQ256)
        rgb_embed = self.rgb_embed(tf_decoder_output_BQ256)
        B, Q, C4 = rgb_embed.shape
        rgb_embed = rgb_embed.reshape(B,Q,4,int(C4/4))
        
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed_BQ256, mask_features_B256hw)

        
        

        deep_super = True
        if ii == 0 and deep_super: # /8 res
            # outputs_mask_BQhw = F.interpolate(outputs_mask, scale_factor=0.5, mode="nearest")
            # outputs_m2a_BQhw = self.convs['m2a'](outputs_mask_BQhw)

            mask_features_B256hw = F.interpolate(mask_features_B256hw, scale_factor=0.5, mode="nearest") # /4 -> /8
            cat_feat_B768hw = torch.cat([encoder_feat['res3'],mask_features_B256hw],1)
            cat_feat_B256hw = self.convs['0_1'](cat_feat_B768hw) # 512+256 -> 256
            RGB_feat_BQ4hw = torch.einsum("bqtc,bchw->bqthw", rgb_embed, cat_feat_B256hw)
                
            if self.use_feat_mask:
                _, _, _, this_H, this_W = RGB_feat_BQ4hw.shape
                feat_mask = outputs_mask.sigmoid()
                feat_mask_BQhw = F.adaptive_avg_pool2d(feat_mask, (this_H, this_W))
                feat_mask_BQ3hw = feat_mask_BQhw[:,:,None,:,:].repeat(1,1,3,1,1)

                RGB_feat_BQ4hw = RGB_feat_BQ4hw * feat_mask_BQ3hw
                RGB_feat_BQx3hw = RGB_feat_BQ4hw.reshape(B,Q*3,this_H,this_W)
                output_rgb_BQx3hw = self.convs['paint'](RGB_feat_BQx3hw) # Qx3 -> Qx3
                outputs_rgb_BQ3hw = output_rgb_BQx3hw.reshape(B,Q,3,this_H,this_W)
            else:
                outputs_rgb_BQ3hw = RGB_feat_BQ4hw
                
            outputs_rgba = RGB_feat_BQ4hw
            # outputs_rgba = torch.cat([outputs_rgb_BQ3hw,outputs_m2a_BQhw[:,:,None,:,:]], dim=2)
        elif ii == 1 and deep_super: # /4 res
            # outputs_mask_BQhw = F.interpolate(outputs_mask, scale_factor=1, mode="nearest")
            # outputs_m2a_BQhw = self.convs['m2a'](outputs_mask_BQhw)

            mask_features_B256hw = F.interpolate(mask_features_B256hw, scale_factor=1, mode="nearest") # /4 -> /4
            cat_feat_B512hw = torch.cat([encoder_feat['res2'],mask_features_B256hw],1)
            cat_feat_B256hw = self.convs['1_1'](cat_feat_B512hw) # 256+256 -> 256
            RGB_feat_BQ4hw = torch.einsum("bqtc,bchw->bqthw", rgb_embed, cat_feat_B256hw)
            
            if self.use_feat_mask:
                _, _, _, this_H, this_W = RGB_feat_BQ4hw.shape
                feat_mask = outputs_mask.sigmoid()
                feat_mask_BQhw = F.adaptive_avg_pool2d(feat_mask, (this_H, this_W))
                feat_mask_BQ3hw = feat_mask_BQhw[:,:,None,:,:].repeat(1,1,3,1,1)

                RGB_feat_BQ4hw = RGB_feat_BQ4hw * feat_mask_BQ3hw
                RGB_feat_BQx3hw = RGB_feat_BQ4hw.reshape(B,Q*3,this_H,this_W)
                output_rgb_BQx3hw = self.convs['paint'](RGB_feat_BQx3hw) # Qx3 -> Qx3
                outputs_rgb_BQ3hw = output_rgb_BQx3hw.reshape(B,Q,3,this_H,this_W)
            else:
                outputs_rgb_BQ3hw = RGB_feat_BQ4hw
                
            outputs_rgba = RGB_feat_BQ4hw
            # outputs_rgba = torch.cat([outputs_rgb_BQ3hw,outputs_m2a_BQhw[:,:,None,:,:]], dim=2)
        elif ii == 2 and deep_super: # /2 res
            # outputs_mask_BQhw = F.interpolate(outputs_mask, scale_factor=2, mode="nearest")
            # outputs_m2a_BQhw = self.convs['m2a'](outputs_mask_BQhw)

            mask_features_B256hw = F.interpolate(mask_features_B256hw, scale_factor=2, mode="nearest") # /4 -> /2
            cat_feat_B320hw = torch.cat([encoder_feat['conv1_out'],mask_features_B256hw],1)
            cat_feat_B256hw = self.convs['2_1'](cat_feat_B320hw) # 64+256 -> 256
            RGB_feat_BQ4hw = torch.einsum("bqtc,bchw->bqthw", rgb_embed, cat_feat_B256hw)

            if self.use_feat_mask:
                _, _, _, this_H, this_W = RGB_feat_BQ4hw.shape
                feat_mask = outputs_mask.sigmoid()
                feat_mask_BQhw = F.adaptive_avg_pool2d(feat_mask, (this_H, this_W))
                feat_mask_BQ3hw = feat_mask_BQhw[:,:,None,:,:].repeat(1,1,3,1,1)

                RGB_feat_BQ4hw = RGB_feat_BQ4hw * feat_mask_BQ3hw
                RGB_feat_BQx3hw = RGB_feat_BQ4hw.reshape(B,Q*3,this_H,this_W)
                output_rgb_BQx3hw = self.convs['paint'](RGB_feat_BQx3hw) # Qx3 -> Qx3
                outputs_rgb_BQ3hw = output_rgb_BQx3hw.reshape(B,Q,3,this_H,this_W)
            else:
                outputs_rgb_BQ3hw = RGB_feat_BQ4hw
                
            outputs_rgba = RGB_feat_BQ4hw
            # outputs_rgba = torch.cat([outputs_rgb_BQ3hw,outputs_m2a_BQhw[:,:,None,:,:]], dim=2)
        elif ii == -1: # full res
            # outputs_mask_BQhw = F.interpolate(outputs_mask, scale_factor=4, mode="nearest")
            # outputs_m2a_BQhw = self.convs['m2a'](outputs_mask_BQhw)

            mask_features_B256hw = F.interpolate(mask_features_B256hw, scale_factor=2, mode="nearest") # /4 -> /2
            cat_feat_B320hw = torch.cat([encoder_feat['conv1_out'],mask_features_B256hw],1)
            cat_feat_B256hw = self.convs['-1_1'](cat_feat_B320hw) # 64+256 -> 256
            cat_feat_B256hw = F.interpolate(cat_feat_B256hw, scale_factor=2, mode="nearest") # /2 -> /1
            cat_feat_B256hw = self.convs['-1_2'](cat_feat_B256hw) # 256 -> 256
            RGB_feat_BQ4hw = torch.einsum("bqtc,bchw->bqthw", rgb_embed, cat_feat_B256hw)

            if self.use_feat_mask:
                _, _, _, this_H, this_W = RGB_feat_BQ4hw.shape
                feat_mask = outputs_mask.sigmoid()
                feat_mask_BQhw = F.adaptive_avg_pool2d(feat_mask, (this_H, this_W))
                feat_mask_BQ3hw = feat_mask_BQhw[:,:,None,:,:].repeat(1,1,3,1,1)

                RGB_feat_BQ4hw = RGB_feat_BQ4hw * feat_mask_BQ3hw
                RGB_feat_BQx3hw = RGB_feat_BQ4hw.reshape(B,Q*3,this_H,this_W)
                output_rgb_BQx3hw = self.convs['paint'](RGB_feat_BQx3hw) # Qx3 -> Qx3
                outputs_rgb_BQ3hw = output_rgb_BQx3hw.reshape(B,Q,3,this_H,this_W)
            else:
                outputs_rgb_BQ3hw = RGB_feat_BQ4hw
                
            outputs_rgba = RGB_feat_BQ4hw
            # outputs_rgba = torch.cat([outputs_rgb_BQ3hw,outputs_m2a_BQhw[:,:,None,:,:]], dim=2)
        else:
            outputs_rgba = torch.zeros([1,1])

        


        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, num_heads, Q, H*W] -> [B*num_heads, Q, H*W]
        # 1st: /4 -> /32
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()
        # 1st: B*8, Q, H/32*W/32

        return outputs_class, outputs_mask, attn_mask, outputs_plane_para, outputs_rgba

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_plane_para):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.

        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_plane_para": c}
                for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_plane_para[:-1])
            ]
        else:
            return [
                {"pred_masks": b, "pred_plane_para": c}
                for b, c in zip(outputs_seg_masks[:-1], outputs_plane_para[:-1])
            ]
            # return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
