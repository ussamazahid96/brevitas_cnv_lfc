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

def paddedSize(inp, padTo):
    if inp % padTo == 0: 
        return inp
    else:
        return inp + padTo - (inp % padTo)
        
# return val to nearest multiple of pad
def padTo(val, pad):
    rem = val % pad
    return val if rem == 0 else (val + pad - rem)

# the quantization function
def quantize(x, integer, fract):
    bits=integer+fract
    if (bits==1):
        return(binarize(x))
    n = float(2**fract) # GIULIO ADD CLIP
    return np.floor(x * n + 0.5) / n    

# the binarization function, basically the sign encoded as 1 for positive and
# 0 for negative
def binarize(w):
    return np.where(w < 0, 0, 1)


# binarize and pack convolutional layer weights into a matrix and compute
# thresholds from the conv bias and batchnorm parameters
def makeConvBNComplex(weights, bias, beta, gamma, mean, invstd, w_prec, a_prec, usePopCount=False, \
    use_rowmajor=True, numThresBits=16, numThresIntBits=None, interleaveChannels=True):

    APrecision = int(a_prec.sum())
    numOut = weights.shape[0]
    numIn = weights.shape[1]
    k = weights.shape[2]
    # the fanin is used to ensure positive-only threshold
    fanin = numIn * k * k
    if(k != weights.shape[3]):
        raise Exception("Nonsymmetric conv kernels are not yet supported")

    # compute a preliminary threshold from the batchnorm parameters,
    # subtracting the conv bias from the batchnorm mean
    if (APrecision == 1) or (APrecision == 0):
        step = np.zeros(1, dtype=np.float64)
    else:
        # This one make -0.5 and +0.5 with 2 bits
        step = np.linspace(-1,1,num=2**(APrecision-1), endpoint=False, dtype=np.float64) + 1./(2**(a_prec[1]+1)) 
        # step = np.linspace(-1,1,num=2**APrecision-1,endpoint=False) # Equidistant points between -1 and +1 (hardtanh)
        # step = step[1:] # Removing the -1 point for symmetrical quantization - hardtanh
    thresholds = np.zeros((len(step),len(mean)), dtype=np.float64)
    for i in range(len(step)):
        thresholds[i] = (mean - bias) + ((step[i] - beta) / (gamma*invstd))
    # ensure all neurons activate on the "positive" side, so we can use
    # greater-than-threshold activation
    need_flip = np.sign(gamma)
    factor = need_flip if numThresIntBits is None else need_flip * (1 << (numThresBits - numThresIntBits))
    thresholds = factor*thresholds
    thresholds = np.floor(thresholds)
    # turn threshold into "number of 1s" (popcount) instead of signed sum
    if usePopCount:
        thresholds = (fanin + thresholds)/2
    thresholds = thresholds.transpose(1,0).astype(np.int)

    # generating weights
    weights = weights * need_flip.reshape(-1,1,1,1)
    weights = quantize(weights, w_prec[0], w_prec[1])
    if interleaveChannels:
        weights = np.moveaxis(weights, 1, -1)
    weights = weights.reshape((numOut, fanin))
    return (weights, thresholds)

def makeFCBNComplex(weights, bias, beta, gamma, mean, invstd, w_prec, a_prec, usePopCount=False, \
    use_rowmajor=True, numThresBits=16, numThresIntBits=None):
    
    ins = weights.shape[0]
    outs = weights.shape[1]
    APrecision = int(a_prec.sum())
    # compute a preliminary thresholds from the batchnorm parameters
    if (APrecision == 1) or (APrecision == 0):
        step = np.zeros(1, dtype=np.float64)
    else:
        # This one make -0.5 and +0.5 with 2 bits
        step = np.linspace(-1,1,num=2**(APrecision-1), endpoint=False, dtype=np.float64) + 1./(2**(a_prec[1]+1))
        # step = np.linspace(-1,1,num=2**APrecision-1,endpoint=False) # Equidistant points between -1 and +1 (hardtanh)
        # step = step[1:] # Removing the -1 point for symmetrical quantization - hardtanh
    thresholds = np.zeros((len(step),len(mean)), dtype=np.float64)
    for i in range(len(step)):
        thresholds[i] = (mean - bias) + ((step[i] - beta) / (gamma*invstd))
    # ensure all neurons activate on the "positive" side, so we can use
    # greater-than-threshold activation
    need_flip = np.sign(gamma)
    factor = need_flip if numThresIntBits is None else need_flip * (1 << (numThresBits - numThresIntBits))
    thresholds = factor*thresholds
    # turn threshold into "number of 1s" (popcount) instead of signed sum
    if usePopCount:
        thresholds = (ins + thresholds)/2
    # Integer-like threshold
    else:
        thresholds = np.floor(thresholds)
    thresholds = thresholds.transpose(1,0).astype(np.int)

    # generating weights
    weights = weights * need_flip
    weights = quantize(weights, w_prec[0], w_prec[1])
    # note how we change from col major to row major if requested
    if use_rowmajor:
        weights = weights.transpose(1,0)
    return (weights, thresholds)

# ensure no non-binary weight values while packing
def ensureBinary(x):
    temp = np.where(x != 0, 1, x)
    temp = np.where(x != 1, 0, temp)
    if not np.array_equal(x,temp):
        raise Exception("Non-binary values found in BNN weight data")

# Encode the array as a single integer number
# The array contains all the values that has to be encoded
# in a single ap_uint.
def ArrayToAp_uints(array, precision, precFract=0):
    if precision == 1:
        ensureBinary(array)
        datatype = np.int64
    else:
        array = array * (1 << precFract)
        array = np.where(array < 0, array+(1 << precision), array).astype(np.uint64)
        datatype = np.uint64
    factor = 1 << precision*np.arange(array.shape[-1], dtype=datatype)
    val = array.dot(factor)
    return val

def padMatrix(A, T, w_prec, a_prec, numThresBits=16, numThresIntBits=None, pop_count=False, padW=0, padH=0):
    n = A.shape[0]
    s = A.shape[1]
    # ensure number of rows (neurons) is divisable by PE count
    padN = padH - n 
    # ensure number of cols (synapses per neuron) is divisable by SIMD width
    padS = padW - s
    # create padded version of matrix
    # use 1 bits to pad matrix, 0 bits to pad input
    const = 1 if w_prec.sum()==1 else 0
    Ap = np.pad(A, ((0, padN), (0, padS)), 'constant', constant_values=const)
    # pad thresholds
    max_thres = (1 << numThresBits) - 1
    Tp = np.pad(T, ((0, padN), (0, 0)), 'constant', constant_values=max_thres)
    if a_prec.sum()==1:
        Tp = Tp.reshape(-1,)
    
    if numThresIntBits is not None:
	    padS = padS << (numThresBits - numThresIntBits) 
    Tp -= 0 if (pop_count or w_prec.sum() >= 2) else padS
    
    # do saturation
    if numThresIntBits is None:
        saturate_max = (1 << (numThresBits-1))-1
        saturate_min = -(1 << (numThresBits-1))
        Tp = np.clip(Tp, saturate_min, saturate_max)
    
    return (Ap, Tp)


def foldMatrix(A, T, numPE, numSIMD, w_prec):
    # TODO also update threshold memories
    # should only be called internally, and on a matrix that is already padded
    n = A.shape[0]
    s = A.shape[1]
    if n % numPE != 0:
        raise Exception("Matrix height must be multiple of PE count")
    if s % numSIMD != 0:
        raise Exception("Matrix width must be multiple of SIMD width")
    if n != T.shape[0]:
        raise Exception("Number of neurons and thresholds do not match")
    # reshape and copy into PE memories
    neuronsPerPE = n // numPE
    synGroupsPerNeuron = s // numSIMD

    M = A.reshape((n, synGroupsPerNeuron, numSIMD))

    M = ArrayToAp_uints(M, int(w_prec.sum()), w_prec[1])
    
    tempw = np.split(M, neuronsPerPE, axis=0)
    tempw = np.asarray(tempw)
    tempw = np.split(tempw, synGroupsPerNeuron, axis=-1)
    tempw = np.asarray(tempw).swapaxes(0,2)
    tempw = tempw.reshape(tempw.shape[0], -1)        
    
    tempt = np.split(T, neuronsPerPE, axis=0)
    tempt = np.array(tempt)
    tempt = tempt.swapaxes(0,1)

    return tempw, tempt