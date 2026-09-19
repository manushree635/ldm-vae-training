# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-12-04

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from scipy.special import factorial

from . import math_utils
pointOp = math_utils.pointOp

################################################################################
################################################################################


class SCFpyr_PyTorch(object):
    '''
    This is a modified version of buildSFpyr, that constructs a
    complex-valued steerable pyramid  using Hilbert-transform pairs
    of filters. Note that the imaginary parts will *not* be steerable.

    Description of this transform appears in: Portilla & Simoncelli,
    International Journal of Computer Vision, 40(1):49-71, Oct 2000.
    Further information: http://www.cns.nyu.edu/~eero/STEERPYR/

    Modified code from the perceptual repository:
      https://github.com/andreydung/Steerable-filter

    This code looks very similar to the original Matlab code:
      https://github.com/LabForComputationalVision/matlabPyrTools/blob/master/buildSCFpyr.m

    Also looks very similar to the original Python code presented here:
      https://github.com/LabForComputationalVision/pyPyrTools/blob/master/pyPyrTools/SCFpyr.py

    '''

    def __init__(self, height=5, nbands=4, scale_factor=2, device=None):
        self.height = height  # including low-pass and high-pass
        self.nbands = nbands  # number of orientation bands
        self.scale_factor = scale_factor
        self.device = torch.device('cpu') if device is None else torch.device(device)
        # Normalize ambiguous "cuda" to an explicit index (usually cuda:0)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda:0')

        # Cache constants
        self.lutsize = 1024
        self.Xcosn = np.pi * np.array(range(-(2*self.lutsize+1), (self.lutsize+2)))/self.lutsize
        self.alpha = (self.Xcosn + np.pi) % (2*np.pi) - np.pi
        self.complex_fact_construct   = np.power(complex(0, -1), self.nbands-1)
        self.complex_fact_reconstruct = np.power(complex(0, 1), self.nbands-1)

        # Cache of precomputed frequency-domain filters keyed by (height, width).
        # The pyramid masks depend only on input size + (self.height, self.nbands,
        # self.scale_factor), never on pixel content, so they can be computed once and
        # reused across every build() call (e.g. every training step).
        self._filter_cache = {}
        
    ################################################################################
    # Construction of Steerable Pyramid

    def _get_filters(self, height, width):
        """Return cached (or freshly built) frequency-domain masks for a given size."""
        key = (int(height), int(width))
        cached = self._filter_cache.get(key)
        if cached is None:
            cached = self._build_filters(height, width)
            self._filter_cache[key] = cached
        return cached

    def _build_filters(self, height, width):
        """Precompute all pyramid masks for a given image size (done once per size)."""
        log_rad, angle = math_utils.prepare_grid(height, width)

        Xrcos, Yrcos = math_utils.rcosFn(1, -0.5)
        Yrcos = np.sqrt(Yrcos)
        YIrcos = np.sqrt(1 - Yrcos**2)

        lo0mask = torch.from_numpy(pointOp(log_rad, YIrcos, Xrcos)).float()[None, :, :].to(self.device)
        hi0mask = torch.from_numpy(pointOp(log_rad, Yrcos, Xrcos)).float()[None, :, :].to(self.device)

        levels = []
        self._build_level_filters(log_rad, angle, Xrcos, Yrcos, self.height - 1, levels)
        return {"lo0mask": lo0mask, "hi0mask": hi0mask, "levels": levels}

    def _build_level_filters(self, log_rad, angle, Xrcos, Yrcos, height, levels):
        """Recursively precompute per-level himask/anglemasks/lomask + subsample indices."""
        if height <= 1:
            return  # terminal low-pass needs no mask

        Xrcos = Xrcos - np.log2(self.scale_factor)

        himask = torch.from_numpy(pointOp(log_rad, Yrcos, Xrcos)).float()[None, :, :].to(self.device)

        order = self.nbands - 1
        const = np.power(2, 2*order) * np.square(factorial(order)) / (self.nbands * factorial(2*order))
        Ycosn = 2*np.sqrt(const) * np.power(np.cos(self.Xcosn), order) * (np.abs(self.alpha) < np.pi/2)

        anglemasks = []
        for b in range(self.nbands):
            anglemask = pointOp(angle, Ycosn, self.Xcosn + np.pi*b/self.nbands)
            anglemasks.append(torch.from_numpy(anglemask).float()[None, :, :].to(self.device))

        dims = np.array(log_rad.shape)
        low_ind_start = (np.ceil((dims+0.5)/2) - np.ceil((np.ceil((dims-0.5)/2)+0.5)/2)).astype(int)
        low_ind_end   = (low_ind_start + np.ceil((dims-0.5)/2)).astype(int)

        nlog_rad = log_rad[low_ind_start[0]:low_ind_end[0], low_ind_start[1]:low_ind_end[1]]
        nangle = angle[low_ind_start[0]:low_ind_end[0], low_ind_start[1]:low_ind_end[1]]

        YIrcos = np.abs(np.sqrt(1 - Yrcos**2))
        lomask = torch.from_numpy(pointOp(nlog_rad, YIrcos, Xrcos)).float()[None, :, :].to(self.device)

        levels.append(
            {
                "himask": himask,
                "anglemasks": anglemasks,
                "low_ind_start": low_ind_start,
                "low_ind_end": low_ind_end,
                "lomask": lomask,
            }
        )

        self._build_level_filters(nlog_rad, nangle, Xrcos, Yrcos, height - 1, levels)

    def build(self, im_batch):
        ''' Decomposes a batch of images into a complex steerable pyramid. 
        The pyramid typically has ~4 levels and 4-8 orientations. 
        
        Args:
            im_batch (torch.Tensor): Batch of images of shape [N,C,H,W]
        
        Returns:
            pyramid: list containing torch.Tensor objects storing the pyramid
        '''
        
        # torch.device('cuda') and torch.device('cuda:0') compare unequal; treat them as compatible.
        if im_batch.device.type != self.device.type:
            raise AssertionError('Devices invalid (pyr = {}, batch = {})'.format(self.device, im_batch.device))
        if self.device.type == 'cuda' and (self.device.index is not None) and (im_batch.device.index is not None):
            assert im_batch.device.index == self.device.index, 'CUDA devices invalid (pyr = {}, batch = {})'.format(self.device, im_batch.device)
        assert im_batch.dtype == torch.float32, 'Image batch must be torch.float32'
        assert im_batch.dim() == 4, 'Image batch must be of shape [N,C,H,W]'
        assert im_batch.shape[1] == 1, 'Second dimension must be 1 encoding grayscale image'

        im_batch = im_batch.squeeze(1)  # flatten channels dim
        height, width = im_batch.shape[1], im_batch.shape[2]
        
        # Check whether image size is sufficient for number of levels
        if self.height > int(np.floor(np.log2(min(width, height))) - 2):
            raise RuntimeError('Cannot build {} levels, image too small.'.format(self.height))

        # Masks depend only on size + pyramid params: compute once, reuse every step.
        filters = self._get_filters(height, width)

        # Fourier transform (2D) and shifting
        batch_dft = torch.fft.fft2(im_batch)
        batch_dft = math_utils.batch_fftshift2d(batch_dft)

        # Low-pass
        lo0dft = batch_dft * filters["lo0mask"]

        # Start recursively building the pyramids
        coeff = self._build_levels_cached(lo0dft, filters["levels"], 0)

        # High-pass
        hi0dft = batch_dft * filters["hi0mask"]
        hi0 = math_utils.batch_ifftshift2d(hi0dft)
        hi0 = torch.fft.ifft2(hi0)
        hi0_real = hi0.real
        coeff.insert(0, hi0_real)
        return coeff

    def _build_levels_cached(self, lodft, levels, level_idx):
        """Same recursion as _build_levels but consuming precomputed masks."""
        if level_idx >= len(levels):
            lo0 = math_utils.batch_ifftshift2d(lodft)
            lo0 = torch.fft.ifft2(lo0)
            return [lo0.real]

        lvl = levels[level_idx]
        himask = lvl["himask"]

        orientations = []
        for b in range(self.nbands):
            banddft = lodft * lvl["anglemasks"][b] * himask
            banddft = banddft * self.complex_fact_construct
            band = math_utils.batch_ifftshift2d(banddft)
            band = torch.fft.ifft2(band)
            orientations.append(band)

        s = lvl["low_ind_start"]
        e = lvl["low_ind_end"]
        lodft = lodft[:, s[0]:e[0], s[1]:e[1]]
        lodft = lvl["lomask"] * lodft

        coeff = self._build_levels_cached(lodft, levels, level_idx + 1)
        coeff.insert(0, orientations)
        return coeff

    def _build_levels(self, lodft, log_rad, angle, Xrcos, Yrcos, height):
        
        if height <= 1:

            # Low-pass
            lo0 = math_utils.batch_ifftshift2d(lodft)
            lo0 = torch.fft.ifft2(lo0)
            lo0_real = lo0.real
            coeff = [lo0_real]

        else:
            
            Xrcos = Xrcos - np.log2(self.scale_factor)

            ####################################################################
            ####################### Orientation bandpass #######################
            ####################################################################

            himask = pointOp(log_rad, Yrcos, Xrcos)
            himask = torch.from_numpy(himask).float()[None, :, :].to(self.device)

            order = self.nbands - 1
            const = np.power(2, 2*order) * np.square(factorial(order)) / (self.nbands * factorial(2*order))
            Ycosn = 2*np.sqrt(const) * np.power(np.cos(self.Xcosn), order) * (np.abs(self.alpha) < np.pi/2) # [n,]

            # Loop through all orientation bands
            orientations = []
            for b in range(self.nbands):

                anglemask = pointOp(angle, Ycosn, self.Xcosn + np.pi*b/self.nbands)
                anglemask = torch.from_numpy(anglemask).float()[None, :, :].to(self.device)

                # Bandpass filtering                
                banddft = lodft * anglemask * himask

                # Multiply with complex phase factor
                banddft = banddft * self.complex_fact_construct

                band = math_utils.batch_ifftshift2d(banddft)
                band = torch.fft.ifft2(band)
                orientations.append(band)

            ####################################################################
            ######################## Subsample lowpass #########################
            ####################################################################

            # Don't consider batch_size and imag/real dim
            dims = np.array(lodft.shape[1:3])  

            # Both are tuples of size 2
            low_ind_start = (np.ceil((dims+0.5)/2) - np.ceil((np.ceil((dims-0.5)/2)+0.5)/2)).astype(int)
            low_ind_end   = (low_ind_start + np.ceil((dims-0.5)/2)).astype(int)

            # Subsampling indices
            log_rad = log_rad[low_ind_start[0]:low_ind_end[0],low_ind_start[1]:low_ind_end[1]]
            angle = angle[low_ind_start[0]:low_ind_end[0],low_ind_start[1]:low_ind_end[1]]

            # Actual subsampling
            lodft = lodft[:, low_ind_start[0]:low_ind_end[0], low_ind_start[1]:low_ind_end[1]]

            # Filtering
            YIrcos = np.abs(np.sqrt(1 - Yrcos**2))
            lomask = pointOp(log_rad, YIrcos, Xrcos)
            lomask = torch.from_numpy(lomask).float()[None, :, :].to(self.device)

            # Convolution in spatial domain
            lodft = lomask * lodft

            ####################################################################
            ####################### Recursion next level #######################
            ####################################################################

            coeff = self._build_levels(lodft, log_rad, angle, Xrcos, Yrcos, height-1)
            coeff.insert(0, orientations)

        return coeff

    ############################################################################
    ########################### RECONSTRUCTION #################################
    ############################################################################

    def reconstruct(self, coeff):

        if self.nbands != len(coeff[1]):
            raise Exception("Unmatched number of orientations")

        height, width = coeff[0].shape[1], coeff[0].shape[2]
        log_rad, angle = math_utils.prepare_grid(height, width)

        Xrcos, Yrcos = math_utils.rcosFn(1, -0.5)
        Yrcos  = np.sqrt(Yrcos)
        YIrcos = np.sqrt(np.abs(1 - Yrcos**2))

        lo0mask = pointOp(log_rad, YIrcos, Xrcos)
        hi0mask = pointOp(log_rad, Yrcos, Xrcos)

        lo0mask = torch.from_numpy(lo0mask).float()[None, :, :].to(self.device)
        hi0mask = torch.from_numpy(hi0mask).float()[None, :, :].to(self.device)

        # Start recursive reconstruction
        tempdft = self._reconstruct_levels(coeff[1:], log_rad, Xrcos, Yrcos, angle)

        hidft = torch.fft.fft2(coeff[0])
        hidft = math_utils.batch_fftshift2d(hidft)

        outdft = tempdft * lo0mask + hidft * hi0mask

        reconstruction = math_utils.batch_ifftshift2d(outdft)
        reconstruction = torch.fft.ifft2(reconstruction)
        reconstruction = reconstruction.real

        return reconstruction

    def _reconstruct_levels(self, coeff, log_rad, Xrcos, Yrcos, angle):

        if len(coeff) == 1:
            dft = torch.fft.fft2(coeff[0])
            dft = math_utils.batch_fftshift2d(dft)
            return dft

        Xrcos = Xrcos - np.log2(self.scale_factor)

        ####################################################################
        ####################### Orientation Residue ########################
        ####################################################################

        himask = pointOp(log_rad, Yrcos, Xrcos)
        himask = torch.from_numpy(himask).float()[None, :, :].to(self.device)

        lutsize = 1024
        Xcosn = np.pi * np.array(range(-(2*lutsize+1), (lutsize+2)))/lutsize
        order = self.nbands - 1
        const = np.power(2, 2*order) * np.square(factorial(order)) / (self.nbands * factorial(2*order))
        Ycosn = np.sqrt(const) * np.power(np.cos(Xcosn), order)

        orientdft = torch.zeros_like(coeff[0][0])
        for b in range(self.nbands):

            anglemask = pointOp(angle, Ycosn, Xcosn + np.pi * b/self.nbands)
            anglemask = torch.from_numpy(anglemask).float()[None, :, :].to(self.device)

            banddft = torch.fft.fft2(coeff[0][b])
            banddft = math_utils.batch_fftshift2d(banddft)

            banddft = banddft * anglemask * himask
            banddft = banddft * self.complex_fact_reconstruct

            orientdft = orientdft + banddft

        ####################################################################
        ########## Lowpass component are upsampled and convoluted ##########
        ####################################################################
        
        dims = np.array(coeff[0][0].shape[1:3])
        
        lostart = (np.ceil((dims+0.5)/2) - np.ceil((np.ceil((dims-0.5)/2)+0.5)/2)).astype(np.int32)
        loend = lostart + np.ceil((dims-0.5)/2).astype(np.int32)

        nlog_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
        nangle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
        YIrcos = np.sqrt(np.abs(1 - Yrcos**2))
        lomask = pointOp(nlog_rad, YIrcos, Xrcos)
        lomask = torch.from_numpy(lomask).float()[None, :, :].to(self.device)

        ################################################################################

        # Recursive call for image reconstruction        
        nresdft = self._reconstruct_levels(coeff[1:], nlog_rad, Xrcos, Yrcos, nangle)

        resdft = torch.zeros_like(coeff[0][0]).to(self.device)
        resdft[:, lostart[0]:loend[0], lostart[1]:loend[1]] = nresdft * lomask

        return resdft + orientdft
