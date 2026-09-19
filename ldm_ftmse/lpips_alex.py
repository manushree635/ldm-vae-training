from __future__ import annotations
"""AlexNet-trunk LPIPS, drop-in for taming's VGG16 LPIPS.

Upstream's LPIPSWithDiscriminator hardcodes `LPIPS()` from
taming.modules.losses.lpips, which is welded to VGG16: chns [64,128,256,512,512],
five vgg16.features slices, lin weights from the `vgg_lpips` checkpoint. There is
no switch. This module mirrors that class exactly and swaps only the trunk, so
everything downstream -- the [-1,1] input convention, the ScalingLayer shift, the
unit-normalise / squared-diff / 1x1-conv / spatial-average chain, and the
[N,1,1,1] output shape -- is bit-for-bit the same computation on a different
feature extractor.

WHAT DIFFERS FROM VGG:
  chns      [64,128,256,512,512]  ->  [64,192,384,256,256]
  slices    vgg16.features 0:4,4:9,9:16,16:23,23:30
            alexnet.features 0:2,2:5,5:8,8:10,10:12
  lin ckpt  taming's `vgg_lpips` (get_ckpt_path download)
            -> lpips_weights/alex_lpips_lin.pth, the official
               richzhang/PerceptualSimilarity v0.1 alex.pth, lifted from
               venv_train's lpips package. It holds ONLY the five 1x1-conv
               lin weights (6 KB); the trunk comes from torchvision's
               pretrained alexnet, pre-downloaded into TORCH_HOME so the DDP
               ranks do not race on first use.

The lin state_dict keys are `lin{k}.model.1.weight`, which already match taming's
NetLinLayer layout (`model = Sequential(Dropout, Conv2d)`, so `.model.1` is the
conv) -- hence use_dropout=True is required for the indices to line up, exactly
as taming defaults.

WHY ALEXNET AT ALL: Zhang et al. 2018 report alex as the best-performing LPIPS
backbone on their 2AFC human-judgement benchmark despite being the smallest, and
it is the default in the reference implementation. VGG is what LDM inherited, not
what was measured to be better. This makes the trunk an ablatable choice rather
than an assumption.
"""

import os

import torch
import torch.nn as nn
from torchvision import models

from taming.modules.losses.lpips import (
    ScalingLayer,
    NetLinLayer,
    normalize_tensor,
    spatial_average,
)

_LIN_CKPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "lpips_weights", "alex_lpips_lin.pth",
)


class alexnet(nn.Module):
    """torchvision alexnet.features cut at the five ReLUs LPIPS taps.

    Slice boundaries are the reference implementation's
    (lpips/pretrained_networks.py): 0:2, 2:5, 5:8, 8:10, 10:12.
    """

    def __init__(self, requires_grad=False, pretrained=True):
        super().__init__()
        feats = models.alexnet(pretrained=pretrained).features
        self.slice1, self.slice2 = nn.Sequential(), nn.Sequential()
        self.slice3, self.slice4, self.slice5 = nn.Sequential(), nn.Sequential(), nn.Sequential()
        self.N_slices = 5
        for x in range(0, 2):
            self.slice1.add_module(str(x), feats[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), feats[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), feats[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), feats[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), feats[x])
        if not requires_grad:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, X):
        h1 = self.slice1(X)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return (h1, h2, h3, h4, h5)


class LPIPSAlex(nn.Module):
    """taming's LPIPS with an AlexNet trunk. Same forward contract."""

    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 192, 384, 256, 256]          # alexnet taps
        self.net = alexnet(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for p in self.parameters():
            p.requires_grad = False

    def load_from_pretrained(self, ckpt=_LIN_CKPT):
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                "AlexNet LPIPS lin weights not found at {}. Copy them from the "
                "lpips package: venv_train/lib/python3.12/site-packages/lpips/"
                "weights/v0.1/alex.pth".format(ckpt))
        sd = torch.load(ckpt, map_location="cpu")
        missing, unexpected = self.load_state_dict(sd, strict=False)
        # Only the trunk should be "missing" -- it comes from torchvision, not
        # this checkpoint. Anything unexpected means the layout drifted.
        if unexpected:
            raise RuntimeError("unexpected keys in {}: {}".format(ckpt, unexpected))
        lin_missing = [k for k in missing if k.startswith("lin")]
        if lin_missing:
            raise RuntimeError("lin weights absent from {}: {}".format(ckpt, lin_missing))
        print("loaded AlexNet LPIPS lin weights from {}".format(ckpt))

    def forward(self, input, target):
        in0, in1 = self.scaling_layer(input), self.scaling_layer(target)
        outs0, outs1 = self.net(in0), self.net(in1)
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        val = None
        for kk in range(len(self.chns)):
            f0, f1 = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            d = (f0 - f1) ** 2
            r = spatial_average(lins[kk].model(d), keepdim=True)
            val = r if val is None else val + r
        return val
