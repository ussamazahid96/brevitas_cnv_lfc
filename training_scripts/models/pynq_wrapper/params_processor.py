# Copyright (c) 2020, Xilinx, Inc.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions are met:

# 1.  Redistributions of source code must retain the above copyright notice, 
#   this list of conditions and the following disclaimer.

# 2.  Redistributions in binary form must reproduce the above copyright 
#   notice, this list of conditions and the following disclaimer in the 
#   documentation and/or other materials provided with the distribution.

# 3.  Neither the name of the copyright holder nor the names of its 
#   contributors may be used to endorse or promote products derived from 
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# 
# Adapted from https://github.com/ussamazahid96/BNN-PYNQ/blob/master/bnn/src/training/finnthesizer.py
#

import numpy as np
from .helpers import *


def model_params_to_bin_converter(layer_idx, conv_layers, numLayers, layer_param_dic, layer_hw_dic, res_dic):
    if layer_idx >= conv_layers:
        fcbn_params_to_bin_converter(layer_idx, numLayers, layer_param_dic, layer_hw_dic, res_dic)
    else:
        convbn_params_to_bin_converter(layer_idx, numLayers, layer_param_dic, layer_hw_dic, res_dic)

def convbn_params_to_bin_converter(layer_idx, numLayers, layer_param_dic, layer_hw_dic, res_dic):
    (weights, bias, beta, gamma, mean, invstd) = layer_param_dic
    (pe, simd, w_prec, in_prec, out_prec, numThresBits, numThresIntBits) = layer_hw_dic
    
    if (w_prec.sum() == 1) and (in_prec.sum() == 1) and (out_prec.sum() == 1):
        pop_count = True
    else:
        pop_count = False

    (weights, thresholds) = makeConvBNComplex(weights, bias, beta, gamma, mean, invstd, w_prec, out_prec, \
        usePopCount=pop_count, numThresBits=numThresBits, numThresIntBits=numThresIntBits)
    
    # compute the padded width and height
    paddedH = padTo(weights.shape[0], pe)
    paddedW = padTo(weights.shape[1], simd)
    
    weights_padded, thresholds_padded = padMatrix(weights, thresholds, w_prec, out_prec, numThresBits, numThresIntBits, pop_count, paddedW, paddedH)
    weights_tiled, thresholds_tiled = foldMatrix(weights_padded, thresholds_padded, pe, simd, w_prec)
    if out_prec.sum() == 1:
        thresholds_tiled = np.expand_dims(thresholds_tiled, axis=-1)
    res_dic["layer_" + str(layer_idx)] = [weights_tiled, thresholds_tiled]

def fcbn_params_to_bin_converter(layer_idx, numLayers, layer_param_dic, layer_hw_dic, res_dic):
    (weights, bias, beta, gamma, mean, invstd) = layer_param_dic
    (pe, simd, w_prec, in_prec, out_prec, numThresBits, numThresIntBits) = layer_hw_dic
    
    if (out_prec[0] == 0):
        #fake the batchnorm params to use same make functions below
        bias   = np.zeros(weights.shape[1])    
        beta   = np.zeros(weights.shape[1])
        #read gamma in case if it has a negative sign, we have to invert the weights
        gamma  = gamma*np.ones(weights.shape[1], dtype=np.float64)
        mean   = np.ones(weights.shape[1])
        invstd = np.ones(weights.shape[1])

    if (w_prec.sum() == 1) and (in_prec.sum() == 1) and (out_prec.sum() == 1):
        pop_count = True
    else:
        pop_count = False

    (weights, thresholds) = makeFCBNComplex(weights, bias, beta, gamma, mean, invstd, w_prec, out_prec, \
        usePopCount=pop_count, numThresBits=numThresBits, numThresIntBits=numThresIntBits)

    # compute the padded width and height
    paddedH = padTo(weights.shape[0], pe)
    paddedW = padTo(weights.shape[1], simd)
    
    if (layer_idx==0): # for the first layer, we pad to multiple of 64 due to the AXI interface
        paddedW = padTo(weights.shape[1], max(simd, 64))
    if (layer_idx==numLayers-1): # for the last layer, we pad to multiple of 64 due to the AXI interface
        paddedH = padTo(weights.shape[0], max(pe, 64))
    
    weights_padded, thresholds_padded = padMatrix(weights, thresholds, w_prec, out_prec, numThresBits, numThresIntBits, pop_count, paddedW, paddedH)
    weights_tiled, thresholds_tiled = foldMatrix(weights_padded, thresholds_padded, pe, simd, w_prec)
    if out_prec.sum() == 1:
        thresholds_tiled = np.expand_dims(thresholds_tiled, axis=-1)
    res_dic["layer_" + str(layer_idx)] = [weights_tiled, thresholds_tiled]
