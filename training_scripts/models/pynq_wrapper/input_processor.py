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

__author__ = "Ussama Zahid"
__copyright__ = "Copyright 2020, Xilinx"

import numpy as np
from .helpers import paddedSize

bitsPerExtMemWord = 64

def binarizeAndPack(imgs):
	values_paded = paddedSize(imgs.shape[1], bitsPerExtMemWord) - imgs.shape[1]
	imgs = np.where(imgs < 0, False, True).astype(np.bool)
	imgs = np.pad(imgs, ((0,0),(0,values_paded)), 'constant', constant_values=False)
	binImages = np.packbits(imgs, axis=1, bitorder='little').view(np.uint64).reshape(-1,)
	return binImages

def quantizeAndPack(imgs):
	imgs = np.moveaxis(imgs, 1, -1).reshape(imgs.shape[0], -1)
	bytes_paded = (paddedSize(imgs.shape[1]*8, bitsPerExtMemWord) - (imgs.shape[1]*8))//8
	imgs = np.pad(imgs, ((0,0),(0,bytes_paded)), 'constant', constant_values=-1)
	imgs = np.clip(imgs, -1, 1-(2**-7))
	imgs = np.round(imgs*2**7)
	binImages = imgs.astype(np.int8).view(np.uint64)
	binImages = binImages.reshape(-1,)
	return binImages

def interleave_channels(imgs, dim1, dim2):
	imgs = imgs.reshape(imgs.shape[0], -1, dim1*dim2)
	imgs = np.swapaxes(imgs, -1, 1).reshape(imgs.shape[0], -1)
	return imgs


def load_mnist_test_set(path, count=None):
	imgs_file = path + "/t10k-images-idx3-ubyte"
	label_file = path + "/t10k-labels-idx1-ubyte"
	
	with open(imgs_file, 'rb') as f:
		magic_number = int.from_bytes(f.read(4), byteorder="big")
		image_count = int.from_bytes(f.read(4), byteorder="big")
		dim = int.from_bytes(f.read(4), byteorder="big")
		dim = int.from_bytes(f.read(4), byteorder="big")
		imgs = np.frombuffer(f.read(), dtype=np.uint8)
		imgs = imgs.reshape(image_count, dim*dim)

	with open(label_file, 'rb') as f:
		magic_number = int.from_bytes(f.read(4), byteorder="big")
		label_count = int.from_bytes(f.read(4), byteorder="big")
		labels = np.frombuffer(f.read(), dtype=np.uint8)

	if count is None:
		count = image_count
		
	return imgs[:count], labels[:count]


def load_cifar10_test_set(path, count=None):
    test_batch = path + "/test_batch"
    with open(test_batch, "rb") as f:
        dict = pickle.load(f, encoding='bytes')
        imgs, labels = dict[b'data'], np.asarray(dict[b'labels'], dtype=np.uint8)
        image_count = imgs.shape[0]
    if count is None:
        count = image_count
    return imgs[:count], labels[:count]
