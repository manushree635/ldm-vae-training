from __future__ import annotations
"""DINOv2 feature-distillation term on top of the GAN/steerable loss, ported from
distillation_nerc (dino.py + losses.py: prepare_dino_input, extract_dino_features,
compute_dino_loss, _feature_cosine_loss -- copied verbatim, only regrouped here).

    loss = <LPIPSWithDiscriminatorSteerable, unchanged> + dino_weight * scale * dino

With steerable_weight 0 the base is exactly the ft-MSE control (mctrl: L1 + LPIPS +
pretrained discriminator), so an arm built on this class is "ctrl + DINO" and nothing
else. The term is added AFTER the base forward, like the steerable term, so upstream's
adaptive discriminator weight (||grad nll|| / ||grad g||) is untouched by it.

UNITS. nerc's loss is mean-reduced: 1 x recon + 0.3 x LPIPS + 1 x DINO, all O(1e-2..1e-1).
Upstream's nll here is a per-image SUM over 3*256*256 elements, so a raw cosine loss
would be ~1e-6 of the objective. dino_normalize: nll_units multiplies the term by
inputs[0].numel() -- the same convention as steerable_normalize -- so dino_weight 1.0
means one unit of DINO per unit of per-pixel recon, as in nerc.

REFERENCE. dino_reference: teacher uses the 512 teacher-VAE reconstruction stashed by
AutoencoderKLSteerableEMA (nerc's use_gt_for_image_losses: false); gt uses the input.
Both sides are bicubic-resized to dino_image_size before DINO, so 512 vs 256 is fine.

THE BACKBONE IS NOT A SUBMODULE. It is held in a plain list, so it is absent from
state_dict (checkpoints and resumes unchanged in structure, export unaffected), from
LitEma, and from DDP's parameter set. It is moved to the batch's device on first use.
It needs transformers >= 4.31 (Dinov2Model); venv_ldm ships 4.19, so launch with
EXTRA_PYTHONPATH=<ldm_official>/pylibs/tf440 (transformers 4.40.2, verified to match
4.44.2 on the same input).
"""

import torch
import torch.nn.functional as F

from .losses_steerable import LPIPSWithDiscriminatorSteerable

_DINO_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_DINO_IMAGENET_STD = (0.229, 0.224, 0.225)


def load_dino_model(model_name: str):
    try:
        from transformers import AutoModel
        try:
            m = AutoModel.from_pretrained(model_name, local_files_only=True)
        except OSError:
            m = AutoModel.from_pretrained(model_name)
    except (ImportError, ValueError, KeyError) as e:
        import transformers
        raise RuntimeError(
            f"cannot load {model_name} with transformers {transformers.__version__} "
            "(DINOv2 needs >= 4.31). Launch with EXTRA_PYTHONPATH=<ldm_official>/pylibs/tf440.") from e
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def prepare_dino_input(images: torch.Tensor, image_size: int) -> torch.Tensor:
    images = (images.clamp(-1.0, 1.0) + 1.0) / 2.0
    if images.shape[-2:] != (image_size, image_size):
        images = F.interpolate(images, size=(image_size, image_size), mode="bicubic",
                               align_corners=False)
    mean = images.new_tensor(_DINO_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(_DINO_IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def _feature_cosine_loss(teacher_feat, student_feat):
    t = F.normalize(teacher_feat.reshape(teacher_feat.size(0), -1), p=2, dim=1)
    s = F.normalize(student_feat.reshape(student_feat.size(0), -1), p=2, dim=1)
    return 1.0 - F.cosine_similarity(t, s, dim=1).mean()


def _dispatch_feature_loss(method, teacher_feat, student_feat):
    method = (method or "cosine").strip().lower()
    if method == "cosine":
        return _feature_cosine_loss(teacher_feat, student_feat)
    if method == "mse":
        return F.mse_loss(student_feat, teacher_feat)
    if method == "l1":
        return F.l1_loss(student_feat, teacher_feat)
    raise ValueError(f"Unknown feature distillation method: {method!r}")


def extract_dino_features(dino_model, images, image_size, layer_indices, *, requires_grad):
    num_layers = len(dino_model.encoder.layer)
    if layer_indices == "all":
        selected = list(range(num_layers + 1))
    else:
        total = num_layers + 1
        selected = sorted({(total + i if i < 0 else i) for i in layer_indices if -total <= i < total})

    pixel_values = prepare_dino_input(images, image_size)
    if requires_grad:
        outputs = dino_model(pixel_values, output_hidden_states=True)
    else:
        with torch.no_grad():
            outputs = dino_model(pixel_values, output_hidden_states=True)

    patch_size = dino_model.config.patch_size
    grid = image_size // patch_size
    num_prefix_tokens = 1 + int(getattr(dino_model.config, "num_register_tokens", 0) or 0)
    patch_features, cls_features = {}, {}
    for idx in selected:
        hs = outputs.hidden_states[idx]
        cls_features[idx] = hs[:, 0]
        patch_tokens = hs[:, num_prefix_tokens:]
        b, n, c = patch_tokens.shape
        patch_features[idx] = patch_tokens.transpose(1, 2).reshape(b, c, grid, grid)
    return patch_features, cls_features


def compute_dino_loss(dino_model, teacher_image, student_image, *, image_size=224,
                      method="cosine", layer_indices=(-1,), cls_weight=1.0,
                      patch_weight=1.0, temperature=1.0):
    teacher_patch, teacher_cls = extract_dino_features(
        dino_model, teacher_image, image_size, layer_indices, requires_grad=False)
    student_patch, student_cls = extract_dino_features(
        dino_model, student_image, image_size, layer_indices, requires_grad=True)
    common = sorted(set(teacher_patch) & set(student_patch))
    if not common:
        zero = torch.tensor(0.0, device=student_image.device)
        return zero, zero, zero
    patch_losses, cls_losses = [], []
    for idx in common:
        patch_losses.append(_dispatch_feature_loss(
            method, teacher_patch[idx].detach(), student_patch[idx]) / (temperature ** 2))
        cls_losses.append(_dispatch_feature_loss(
            method, teacher_cls[idx].detach(), student_cls[idx]) / (temperature ** 2))
    dino_patch_loss = sum(patch_losses) / len(patch_losses)
    dino_cls_loss = sum(cls_losses) / len(cls_losses)
    return (patch_weight * dino_patch_loss + cls_weight * dino_cls_loss,
            dino_patch_loss, dino_cls_loss)


class LPIPSWithDiscriminatorSteerableDINO(LPIPSWithDiscriminatorSteerable):
    def __init__(self, *args,
                 dino_weight=0.0,
                 dino_model_name="facebook/dinov2-small",
                 dino_image_size=224,
                 dino_method="cosine",
                 dino_layer_indices=(-1,),
                 dino_cls_weight=1.0,
                 dino_patch_weight=1.0,
                 dino_temperature=1.0,
                 dino_normalize="nll_units",            # "nll_units" | "none"
                 dino_reference="teacher",              # "teacher" | "gt"
                 **kwargs):
        super().__init__(*args, **kwargs)
        if dino_normalize not in ("nll_units", "none"):
            raise ValueError("dino_normalize must be 'nll_units' or 'none'")
        if dino_reference not in ("teacher", "gt"):
            raise ValueError("dino_reference must be 'teacher' or 'gt'")
        if (getattr(self, "nll_reduction", "sum") == "mean" and dino_normalize == "nll_units"
                and float(dino_weight) != 0.0):
            raise ValueError("nll_reduction='mean' requires dino_normalize='none'")
        self.dino_weight = float(dino_weight)
        self.dino_image_size = int(dino_image_size)
        self.dino_method = str(dino_method)
        self.dino_layer_indices = (dino_layer_indices if dino_layer_indices == "all"
                                   else [int(i) for i in dino_layer_indices])
        self.dino_cls_weight = float(dino_cls_weight)
        self.dino_patch_weight = float(dino_patch_weight)
        self.dino_temperature = float(dino_temperature)
        self.dino_normalize = dino_normalize
        self.dino_reference = dino_reference
        self._dino = [load_dino_model(dino_model_name) if self.dino_weight != 0.0 else None]
        if self._dino[0] is not None:
            n = sum(p.numel() for p in self._dino[0].parameters())
            print(f"[dino] {dino_model_name} ({n/1e6:.1f}M, frozen, not a submodule) "
                  f"weight={self.dino_weight} {dino_normalize} ref={dino_reference} "
                  f"size={self.dino_image_size} layers={self.dino_layer_indices} {self.dino_method}")

    def _dino_on(self, device):
        m = self._dino[0]
        if next(m.parameters()).device != device:
            m.to(device)
        return m

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx, global_step,
                *args, **kwargs):
        loss, log = super().forward(inputs, reconstructions, posteriors, optimizer_idx,
                                    global_step, *args, **kwargs)
        if optimizer_idx != 0 or self.dino_weight == 0.0:
            return loss, log
        split = kwargs.get("split", "train")
        ref = self.teacher_input if self.dino_reference == "teacher" else inputs
        if ref is None:
            raise RuntimeError("dino_reference='teacher' but no teacher was stashed; use "
                               "AutoencoderKLSteerableEMA with a FileListSRTeacher* dataset")
        d, dp, dc = compute_dino_loss(
            self._dino_on(reconstructions.device), ref, reconstructions,
            image_size=self.dino_image_size, method=self.dino_method,
            layer_indices=self.dino_layer_indices, cls_weight=self.dino_cls_weight,
            patch_weight=self.dino_patch_weight, temperature=self.dino_temperature)
        scale = float(inputs[0].numel()) if self.dino_normalize == "nll_units" else 1.0
        term = self.dino_weight * scale * d
        loss = loss + term
        key = "{}/total_loss".format(split)
        if key in log:
            log[key] = log[key] + term.detach()
        log["{}/dino_loss".format(split)] = d.detach()          # unit-free, nerc-comparable
        log["{}/dino_patch_loss".format(split)] = dp.detach()
        log["{}/dino_cls_loss".format(split)] = dc.detach()
        log["{}/dino_term".format(split)] = term.detach()       # loss units
        return loss, log
