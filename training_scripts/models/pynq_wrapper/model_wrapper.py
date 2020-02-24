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

import os
import time
import cffi
import torch
import pickle
import numpy as np
from .helpers import paddedSize
from abc import ABC, abstractmethod
from pynq import Overlay, PL, allocate

bitsPerExtMemWord = 64

if os.environ['BOARD'] == 'Ultra96':
    PLATFORM="ultra96"
elif os.environ['BOARD'] == 'Pynq-Z1' or os.environ['BOARD'] == 'Pynq-Z2':
    PLATFORM="pynqZ1-Z2"
else:
    raise RuntimeError("Board not supported")

BNN_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

_ffi = cffi.FFI()

_ffi.cdef("""
void load_layer(unsigned long long *w_arr, unsigned long long *t_arr, unsigned int layer, unsigned int PEs, unsigned int Wtiles, unsigned int Ttiles, unsigned int API, unsigned int addr);
void deinit();
"""
)

_libraries = {}

class model_wrapper(ABC):
    def __init__(self, network, dir_path, download_bitstream=True):
        self.bitstream_name="{0}-{1}.bit".format(network,PLATFORM)
        self.bitstream_path = dir_path + "/bitstreams/" + self.bitstream_name
        self.bitstream_path = self.bitstream_path.replace("//", "/")
        self.overlay = Overlay(self.bitstream_path, download=download_bitstream)
        if PL.bitfile_name != self.bitstream_path:
            raise RuntimeError("Incorrect Overlay loaded")
        self.bbj = self.overlay.BlackBoxJam_0.register_map
        self.base_addr = self.overlay.BlackBoxJam_0.mmio.base_addr
        self.network_name = network
        dllname = "layer_loader.so"
        if dllname not in _libraries:
            _libraries[dllname] = _ffi.dlopen(os.path.join(BNN_ROOT_DIR, dllname))
        self.interface = _libraries[dllname]
        self.accel_input_buffer = None
        self.accel_output_buffer = None
        self.classes = [0,1,2,3,4,5,6,7,8,9]
        
    # function to set weights and activation thresholds of specific network
    def load_parameters(self, params_dic, hw_dic):
        for layer in range(len(params_dic)):
            (weights, thresholds) = params_dic["layer_"+str(layer)]
            (_, _, _, _, out_prec, _, _) = hw_dic["layer_"+str(layer)]
            thresholds = thresholds.astype(np.uint64)

            # python for loops to load params. Very slow!
            # self.load_layer_weights(layer, weights)
            # if out_prec[0] > 0:
            #     self.load_layer_thresholds(layer, thresholds)
            
            # .so driver to load params. ~1000x faster than python for loops!
            (pe, Wtiles) = weights.shape
            (_, Ttiles, num_threshs) = thresholds.shape
            thresholds = thresholds.reshape(-1)
            w_ptr = _ffi.cast("unsigned long long *", weights.ctypes.data)
            t_ptr = _ffi.cast("unsigned long long *", thresholds.ctypes.data)
            self.interface.load_layer(w_ptr, t_ptr, layer, pe, Wtiles, Ttiles, out_prec[0], self.base_addr)
        self.interface.deinit()


    # def load_layer_weights(self, layer, params):
    #     for pe in range(params.shape[0]):
    #         for tile in range(params.shape[1]):
    #             self.set_param(2*layer, pe, tile, 0, int(params[pe][tile]))

    # def load_layer_thresholds(self, layer, params):
    #     for pe in range(params.shape[0]):
    #         for tile in range(params.shape[1]):
    #             for thresh_idx in range(params.shape[2]):
    #                 self.set_param(2*layer+1, pe, tile, thresh_idx, int(params[pe][tile][thresh_idx]))

    # def set_param(self, layer, pe, tile, thresh_idx, param):
    #     self.bbj.doInit = 1

    #     self.bbj.targetLayer = layer
    #     self.bbj.targetMem = pe
    #     self.bbj.targetInd = tile
    #     self.bbj.targetThresh = thresh_idx

    #     self.bbj.val_V_1 = param & 0xffffffff
    #     self.bbj.val_V_2 = (param >> 32) & 0xffffffff
    #     print(param)
    #     self.ExecAccel()

    #     self.bbj.doInit = 0

    def ExecAccel(self):
        self.bbj.CTRL.AP_START = 1
        while not self.bbj.CTRL.AP_DONE:
            pass

    def forward(self, imgs, num_imgs):
        if self.accel_input_buffer is None or self.accel_output_buffer is None:             
            self.allocate_io_buffers(input_shape=imgs.shape, output_shape=(num_imgs*self.psl,))
        np.copyto(self.accel_input_buffer, imgs)
        self.bbj.numReps = num_imgs
        self.ExecAccel()
        predictions = np.copy(np.frombuffer(self.accel_output_buffer, dtype=np.uint64))
        predictions = self.postprocessor(predictions, num_imgs)
        predictions = torch.from_numpy(2*predictions-1).type(torch.FloatTensor)
        return predictions

    @abstractmethod
    def postprocessor(self):
        pass

    def allocate_io_buffers(self, input_shape, output_shape):
        self.accel_input_buffer  = allocate(shape=input_shape, dtype=np.uint64)
        self.accel_output_buffer = allocate(shape=output_shape, dtype=np.uint64)
        np.copyto(self.accel_output_buffer, np.zeros(output_shape, dtype=np.uint64))
        self.bbj.in_V_1 = self.accel_input_buffer.physical_address & 0xffffffff
        self.bbj.in_V_2 = (self.accel_input_buffer.physical_address >> 32) & 0xffffffff
        self.bbj.out_V_1 = self.accel_output_buffer.physical_address & 0xffffffff
        self.bbj.out_V_2 = (self.accel_output_buffer.physical_address >> 32) & 0xffffffff
        self.bbj.doInit = 0


    def free_io_buffers(self):
        del self.accel_input_buffer
        del self.accel_output_buffer
    
    def __del__(self):
        self.free_io_buffers()


class cnvWrapper(model_wrapper):
    def __init__(self, *kargs, **kwargs):
        super(cnvWrapper, self).__init__(*kargs, **kwargs)

        paddedclasses = paddedSize(len(self.classes), bitsPerExtMemWord)
        self.psl = paddedSize(paddedclasses*16, bitsPerExtMemWord) // bitsPerExtMemWord

    def postprocessor(self, predictions, num_imgs):
        predictions = predictions.reshape(num_imgs, -1).view(np.int16)
        predictions = predictions[:,:len(self.classes)]
        max_ = np.expand_dims(np.max(predictions, axis=1), axis=-1)
        predictions = predictions / max_
        return predictions


class lfcWrapper(model_wrapper):
    def __init__(self, *kargs, **kwargs):
        super(lfcWrapper, self).__init__(*kargs, **kwargs)
        self.psl = paddedSize(len(self.classes), bitsPerExtMemWord) // bitsPerExtMemWord

    def postprocessor(self, predictions, num_imgs):
        predictions = np.where(predictions==0, 2**9, predictions)
        predictions = np.log2(predictions).astype(np.uint)
        predictions = predictions[:num_imgs]
        predictions = np.hstack(predictions)
        predictions = np.eye(len(self.classes))[predictions]
        return predictions

