import torch
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, VisionTransformer


class ViTAttribution:
    @staticmethod
    def attribute(model, x, method="rollout", as_regression=False, **kwargs):
        match method:
            case "rollout" | "attention_rollout":
                kwargs.pop("output", None)
                return ViTAttribution.attention_rollout(model, x, **kwargs)
            case "chefer":
                return ViTAttribution.chefer_attribution(model, x, as_regression=as_regression, **kwargs)

    @staticmethod
    @torch.no_grad()
    def get_rollout(
        attentions,
        head_fusion="mean",
        discard_ratio=0.9,
        reshape=True,
        start_from=0,
        mode="original",
        clamp=False,
        clamp_value=0.0,
        num_prefix_tokens=1,
    ):
        eye = torch.eye(attentions[0].size(-1), device=attentions[0].device)
        result = eye
        for attention in attentions[start_from:]:
            if clamp:
                attention = attention.clamp(min=clamp_value)
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            elif head_fusion == "skip":
                attention_heads_fused = attention
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            if mode == "original":
                I = torch.eye(attention_heads_fused.size(-1), device=attention_heads_fused.device)
                a = (attention_heads_fused + 1.0 * I) / 2
                a = a / a.sum(dim=-1)
                result = torch.matmul(a, result)
            else:
                result = result + torch.matmul(attention_heads_fused, result)

        if mode == "chefer":
            result = result / result.sum(dim=-1, keepdim=True) + eye
        result = result[:, 0, num_prefix_tokens:]
        if reshape:
            N = int(result.size(-1) ** 0.5)
            result = result.view(-1, 1, N, N)
        return result

    @staticmethod
    def attention_rollout(model, x, **kwargs):
        assert isinstance(model, VisionTransformer), "Model must be a Vision Transformer"
        assert hasattr(model, "cls_token"), "Model must have classification token"

        attention_matrices = ViTAttribution.get_attention_matrices(model, x)
        assert len(attention_matrices) > 0, "No attention matrices found"
        rollout = ViTAttribution.get_rollout(attention_matrices, num_prefix_tokens=model.num_prefix_tokens, **kwargs)
        return rollout

    @staticmethod
    def chefer_attribution(model, x, as_regression=False, **kwargs):
        assert isinstance(model, VisionTransformer), "Model must be a Vision Transformer"
        assert hasattr(model, "cls_token"), "Model must have classification token"

        attns, grad_attns = ViTAttribution.get_attention_matrices_with_gradients(
            model, x, as_regression=as_regression, output=kwargs.pop("output", None)
        )
        As = [ViTAttribution.avg_heads(a, g) for a, g in zip(attns, grad_attns)]
        return ViTAttribution.get_rollout(As, mode="chefer", num_prefix_tokens=model.num_prefix_tokens, **kwargs)

    @torch.no_grad()
    def get_attention_matrices(model, x):
        hooks = []
        attention_matrices = []

        def hook_fn(module, input, output):
            attention_matrices.append(input[0])

        for n, m in model.named_modules():
            if isinstance(m, Attention):
                m.fused_attn = False
            if "attn_drop" in n:
                hook = m.register_forward_hook(hook_fn)
                hooks.append(hook)

        assert len(hooks) > 0, "No attention layers found"
        model(x)

        for h in hooks:
            h.remove()
        return attention_matrices

    def get_attention_matrices_with_gradients(model, x, output=None, as_regression=False):
        hooks = []
        grad_attention_matrices = []
        attention_matrices = []

        def forward_hook_fn(module, input, output):
            attention_matrices.append(input[0])

        def backward_hook_fn(module, grad_input, grad_output):
            grad_attention_matrices.append(grad_input[0])

        for n, m in model.named_modules():
            if isinstance(m, Attention):
                m.fused_attn = False
            if "attn_drop" in n:
                m.test_name = n
                hook = m.register_forward_hook(forward_hook_fn)
                hooks.append(hook)
                hook = m.register_full_backward_hook(backward_hook_fn)
                hooks.append(hook)

        assert len(hooks) > 0, "No attention layers found"
        pred = model(x)
        
        if output is None:
            if as_regression:
                output = torch.round(pred)
            else:
                output = pred.argmax(dim=-1)
        else:
            if not isinstance(output, torch.Tensor):
                if isinstance(output, list):
                    output = torch.tensor(output, device=pred.device, dtype=pred.dtype)
                elif isinstance(output, int):
                    output = torch.tensor([output], device=pred.device, dtype=pred.dtype)
                    output = output.repeat(pred.size(0))

        if as_regression:
            l = F.mse_loss(pred.float(), output)
        else:
            l = F.cross_entropy(pred, output.long())
        l.backward(retain_graph=True)
        for h in hooks:
            h.remove()
        return attention_matrices, grad_attention_matrices[::-1]

    @staticmethod
    def avg_heads(cam, grad):
        grad = grad / grad.amax(dim=(-2, -1), keepdim=True)
        # cam = cam / cam.amax(dim=(-2, -1), keepdim=True)
        cam = grad * cam
        cam = cam.clamp(min=0)
        return cam
