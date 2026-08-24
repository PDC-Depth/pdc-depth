import os
import torch
import torch.nn as nn
import models.backbones_video as backbones
from models.backbones_video.mylora import Linear as LoraLinear
from models.backbones_video.mylora import DVLinear as DVLinear
from .layers import mark_only_part_as_trainable, mark_adapter_as_trainable, _make_scratch, _make_fusion_block
from .dpt_temporal_MATA import DPTHeadTemporal

    
class endodac_video(nn.Module):
    """Applies low-rank adaptation to a ViT model's image encoder.

    Args:
        backbone_size: size of pretrained Dinov2 choice from: "small", "base", "large", "giant"
        r: rank of LoRA
        image_shape: input image shape, h,w need to be multiplier of 14, default:(224,280)
        lora_layer: which layer we apply LoRA.
    """

    def __init__(self, 
                 backbone_size = "large", 
                 r=4, 
                 image_shape=(224,280), 
                 lora_type="lora",
                 pretrained_path=None,
                 residual_block_indexes=[],
                 include_cls_token=True,
                 use_cls_token=False,
                 use_bn=False,
                 num_frames=1,
                 pe='ape'):
        super(endodac_video, self).__init__()

        assert r > 0
        self.r = r
        self.backbone_size = backbone_size
        self.backbone = {
            "small": backbones.vits.vit_small(input_size=image_shape, residual_block_indexes=residual_block_indexes,
                                              include_cls_token=include_cls_token),
            "large": backbones.vits.vit_large(input_size=image_shape, residual_block_indexes=residual_block_indexes,
                                              include_cls_token=include_cls_token),
        }
        self.backbone_archs = {
            "small": "vits14",
            "large": "vitl14",
        }
        self.intermediate_layers = {
            "small": [2, 5, 8, 11],
            "large": [4, 11, 17, 23],
        }
        self.embedding_dims = {
            "small": 384,
            "large": 1024,
        }
        self.depth_head_features = {
            "small": 64,
            "large": 256,
        }
        self.depth_head_out_channels = {
            "small": [48, 96, 192, 384],
            "large": [256, 512, 1024, 1024],
        }
        self.backbone_arch = self.backbone_archs[self.backbone_size]
        self.embedding_dim = self.embedding_dims[self.backbone_size]
        self.depth_head_feature = self.depth_head_features[self.backbone_size]
        self.depth_head_out_channel = self.depth_head_out_channels[self.backbone_size]
        encoder = self.backbone[self.backbone_size]

        self.image_shape = image_shape
        
        if lora_type != "none":
            for t_layer_i, blk in enumerate(encoder.blocks):
                mlp_in_features = blk.mlp.fc1.in_features
                mlp_hidden_features = blk.mlp.fc1.out_features
                mlp_out_features = blk.mlp.fc2.out_features
                blk.mlp.fc1 = DVLinear(mlp_in_features, mlp_hidden_features, r=self.r, lora_alpha=self.r)
                blk.mlp.fc2 = DVLinear(mlp_hidden_features, mlp_out_features, r=self.r, lora_alpha=self.r)

        self.encoder = encoder
        self.depth_head = DPTHeadTemporal(self.embedding_dim, self.depth_head_feature, use_bn, 
                                          out_channels=self.depth_head_out_channel, use_clstoken=use_cls_token, num_frames=num_frames, pe=pe)

        if pretrained_path is not None:
            pretrained_path = os.path.join(pretrained_path, "{}.pth".format('video_depth_anything_'+self.backbone_arch[:-2]))
            pretrained_dict = torch.load(pretrained_path)
            model_dict = self.state_dict()
            self.load_state_dict(pretrained_dict, strict=False)
            print("load pretrained weight from {}\n".format(pretrained_path))

        mark_only_part_as_trainable(self.encoder)
        mark_only_part_as_trainable(self.depth_head)
        mark_adapter_as_trainable(self.depth_head)
    def forward(self, pixel_values):
        B, T, C, H, W = pixel_values.shape
        pixel_values = pixel_values.view(B * T, C, H, W)
        pixel_values = torch.nn.functional.interpolate(pixel_values, size=self.image_shape, mode="bilinear", align_corners=True)
        pixel_values = pixel_values.view(B, T, C, *self.image_shape)

        h, w = pixel_values.shape[-2:]
        
        features = self.encoder.get_intermediate_layers(pixel_values.flatten(0,1), 4, return_class_token=True)
        patch_h, patch_w = h // 14, w // 14

        disp = self.depth_head(features, patch_h, patch_w, T)

        for disp_key in disp:
            disp_val = disp[disp_key]
            disp_val = disp_val.view(B, T, 1, *disp_val.shape[-2:])
            disp[disp_key] = disp_val

        return disp
