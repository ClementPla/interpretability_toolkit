import torch
import torch.nn.functional as F
from captum.attr import (
    LRP,
    GradientShap,
    GuidedGradCam,
    IntegratedGradients,
    KernelShap,
    LayerDeepLift,
    LayerGradCam,
    Occlusion,
    ShapleyValueSampling,
)

from interpretability_toolkit.utils.tensors import get_last_layer
from interpretability_toolkit.vits.focused_attention import FocusedAttention
from interpretability_toolkit.vits.vits_attrs import ViTAttribution


class Attribution:
    def __init__(
        self,
        model,
        # Focused Attention Parameters
        sampling_points=256,
        with_replacement=True,
        temperature=1.5,
        dilate_tokens=False,
        attribution_method="rollout",
        noise_level=0.25,
    ):
        self.model = model
        layer = get_last_layer(model)
        self.ig = IntegratedGradients(model)
        self.shapley = ShapleyValueSampling(model)
        self.guided_gradcam = GuidedGradCam(model, layer=layer)
        self.lrp = LRP(model)
        self.occlusion = Occlusion(model)
        self.gradientSHAP = GradientShap(model)
        self.kernelSHAP = KernelShap(model)
        self.gradcam = LayerGradCam(model, layer)
        self.deep_lift = LayerDeepLift(model, layer)
        self.focused_attention = FocusedAttention(
            model,
            sampling_points=sampling_points,
            with_replacement=with_replacement,
            temperature=temperature,
            dilate_tokens=dilate_tokens,
            attribution_method=attribution_method,
            noise_level=noise_level,
        )

    @property
    def methods(self):
        return ["ig", "shapley"]

    def attribute(self, input, method, **kwargs):
        baseline = torch.zeros_like(input)
        match method:
            case "ig":
                attr = self.ig.attribute(input, baselines=baseline, **kwargs)
            case "shapley":
                attr = self.shapley.attribute(input, baselines=baseline, **kwargs)
            case "guided_gradcam":
                attr = self.guided_gradcam.attribute(input, **kwargs)
            case "lrp":
                attr = self.lrp.attribute(input, **kwargs)
            case "occlusion":
                if "sliding_window_shapes" not in kwargs:
                    kwargs["sliding_window_shapes"] = (3, 16, 16)
                if "strides" not in kwargs:
                    kwargs["strides"] = 5
                attr = self.occlusion.attribute(input, **kwargs)
            case "gradientSHAP":
                if "n_samples" not in kwargs:
                    kwargs["n_samples"] = 40
                attr = self.gradientSHAP.attribute(input, baselines=baseline, **kwargs)
            case "kernelSHAP":
                if "n_samples" not in kwargs:
                    kwargs["n_samples"] = 40
                attr = self.kernelSHAP.attribute(input, **kwargs)
            case "gradcam":
                attr = self.gradcam.attribute(input, **kwargs)
            case "deep_lift":
                attr = self.deep_lift.attribute(input, baselines=baseline, **kwargs)
            case "attention_rollout" | "rollout":
                attr = ViTAttribution.attribute(
                    self.model, input, method="rollout", **kwargs
                )
            case "focused_attention":
                attr = self.focused_attention.attribute(input, **kwargs)
            case "chefer":
                attr = ViTAttribution.attribute(
                    self.model, input, method="chefer", **kwargs
                )
            case _:
                raise ValueError(f"Unsupported attribution method: {method}")

        if attr.ndim == 3:
            B, _, N = attr.shape
            attr = attr.view(B, 1, int(N**0.5), int(N**0.5))
        attr = F.interpolate(
            attr, input.shape[-2:], mode="bilinear", align_corners=True
        )
        return attr
