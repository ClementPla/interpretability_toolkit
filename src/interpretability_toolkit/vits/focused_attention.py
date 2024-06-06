import math

import kornia as K
import numpy as np
import torch
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from tqdm import tqdm

from interpretability_toolkit.vits.vits_attrs import ViTAttribution
from interpretability_toolkit.vits.utils import _remove_all_forward_hooks
from copy import deepcopy

class DynamicViT(object):
    def __init__(self, model: VisionTransformer) -> None:
        self.model = model
        self.initial_stride = model.patch_embed.proj.stride[0]
        self.dynamic_img_size = model.dynamic_img_size
        self.initial_positional_embedding = torch.nn.Parameter(model.pos_embed.clone())
        self.patch_embed = deepcopy(model.patch_embed)
        
    def __enter__(self):
        self.model.patch_embed.flatten = False
        self.model.patch_embed.output_fmt = "NHWC"
        self.model.dynamic_img_size = True
        return self.model
    def __exit__(self, type, value, traceback):
        print("Restoring the model to its original state.")
        self.model.dynamic_img_size = self.dynamic_img_size
        self.model.patch_embed = self.patch_embed
        self.model.pos_embed = self.initial_positional_embedding
        _remove_all_forward_hooks(self.model)
        
class FocusedAttention:
    def __init__(
        self,
        model: VisionTransformer,
        sampling_points=256,
        with_replacement=True,
        temperature=1.5,
        dilate_tokens=False,
        attribution_method="rollout",
        noise_level=0.25,
    ):
        self.model = model
        self.sampling_points = sampling_points
        self.with_replacement = with_replacement
        self._indices = None
        self._attention_map = None
        self._local_map = None
        self._t = temperature
        self.dilate = dilate_tokens
        self.attribution_method = attribution_method
        self.noise_level = noise_level
    def attribute(self, x, **kwargs):
        return self.focused_attention(x, **kwargs)

    def focused_attention(
        self,
        x,
        output=None,
        **kwargs,
    ):
        initial_stride = self.model.patch_embed.proj.stride[0]
        powers = np.arange(0, np.log2(initial_stride) + 1)
        strides = (2**powers).astype(int)[::-1]
        strides = np.arange(1, initial_stride + 1, 1)[::-1]
        with DynamicViT(self.model) as model:
            if output is not None:
                if not isinstance(output, torch.Tensor):
                    if isinstance(output, list):
                        output = torch.tensor(output, device=x.device, dtype=x.dtype)
                    elif isinstance(output, int):
                        output = torch.tensor([output], device=x.device, dtype=x.dtype)
                        output = output.repeat(x.size(0))
            for i, stride in enumerate(tqdm(strides)):
                model.patch_embed.proj.stride = (stride, stride)
                if i == 0:
                    rollout = ViTAttribution.attribute(
                        model,
                        x,
                        output=output,
                        method=self.attribution_method,
                        reshape=False,
                        **kwargs
                    )
                    self._attention_map = rollout
                else:
                    K = math.ceil(self._t**i)
                    for j in range(K):
                        self.start_one_iteration(
                            x, output=output, **kwargs
                        )
                        if j == 0:
                            lmaps = self._local_map / K
                        else:
                            lmaps += (self._local_map) / K
                    self._attention_map = torch.maximum(self._attention_map, lmaps)
                    self._local_map = None

        N = self._attention_map.size(-1)
        H = W = int(N**0.5)
        rollout = self._attention_map.view(-1, 1, H, W)
        return rollout

    def start_one_iteration(self, x, output=None, **kwargs):
        h = self.model.norm_pre.register_forward_hook(self.drop_patches)
        rollout = ViTAttribution.attribute(
            self.model,
            x,
            output=output,
            method=self.attribution_method,
            reshape=False,
            **kwargs
        )
        self._local_map = self._local_map.to(rollout)
        self._local_map.scatter_reduce_(
            1, self._indices, rollout, reduce="max", include_self=True
        )
        if self.dilate:
            N = self._local_map.size(-1)
            H = W = int(N**0.5)
            self._local_map = self._local_map.view(-1, 1, H, W)
            kernel = self._local_map.new_ones(7, 7)
            self._local_map = K.morphology.dilation(self._local_map, kernel)
            self._local_map = self._local_map.flatten(1)
        h.remove()

    def drop_patches(self, module, input, output):
        B = output.shape[0]
        N = output.shape[1] - self.model.num_prefix_tokens
        H = W = int(N**0.5)
        if self._local_map is None:
            self._local_map = output.new_zeros(B, N, dtype=output.dtype)

        assert self._attention_map is not None, "Attention map is None"
        old_N = self._attention_map.size(-1)
        old_H = old_W = int(old_N**0.5)
        self._attention_map = self._attention_map.view(-1, 1, old_H, old_W)
        if self._attention_map.shape[-2:] != (H, W):
            self._attention_map = F.interpolate(
                self._attention_map, (H, W), mode="bilinear", align_corners=True
            )
            self._attention_map = self._attention_map.view(-1, H * W)
            # self._attention_map = self._attention_map / self._attention_map.sum(dim=-1)
        else:
            self._attention_map = self._attention_map.view(-1, H * W)
        if N < self.sampling_points:
            self._indices = torch.arange(N, device=output.device).repeat(B, 1)
        else:
            sampling_map = self._attention_map - self._attention_map.mean(dim=-1, keepdim=True)
            sampling_map = torch.nan_to_num(sampling_map)
            self._indices = torch.multinomial(
                torch.sigmoid(sampling_map) + self.noise_level, self.sampling_points, replacement=self.with_replacement
            )
            cls_token = output[:, :self.model.num_prefix_tokens]
            output = output[:, self.model.num_prefix_tokens:]
            output = torch.gather(
                output, 1, self._indices.unsqueeze(2).expand(-1, -1, output.size(-1))
            )
            output = torch.cat([cls_token, output], dim=1)
        return output
