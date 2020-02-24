# MIT License
#
# Copyright (c) 2019 Xilinx
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from operator import mul
from functools import reduce

import torch
from brevitas.nn import QuantLinear, QuantHardTanh
from torch.nn import Module, ModuleList, BatchNorm1d, Dropout

from .common import get_quant_linear, get_act_quant, get_quant_type, get_stats_op

from .pynq_wrapper.input_processor import binarizeAndPack

FC_OUT_FEATURES = [1024, 1024, 1024]
INTERMEDIATE_FC_PER_OUT_CH_SCALING = True
LAST_FC_PER_OUT_CH_SCALING = False
IN_DROPOUT = 0.2
HIDDEN_DROPOUT = 0.2
EPS = 1e-5

class LFC(Module):

    def __init__(self, config, num_classes=10, in_ch=1, in_features=(28, 28)):
        super(LFC, self).__init__()
        
        self.config = config
        self.name = "lfcW{}A{}".format(self.config.weight_bit_width, self.config.act_bit_width)
        weight_quant_type = get_quant_type(self.config.weight_bit_width)
        act_quant_type = get_quant_type(self.config.act_bit_width)
        in_quant_type = get_quant_type(self.config.in_bit_width)
        stats_op = get_stats_op(weight_quant_type)

        self.features = ModuleList()
        self.features.append(get_act_quant(self.config.in_bit_width, in_quant_type))
        self.features.append(Dropout(p=IN_DROPOUT))
        in_features = reduce(mul, in_features)
        for out_features in FC_OUT_FEATURES:
            self.features.append(get_quant_linear(in_features=in_features,
                                                  out_features=out_features,
                                                  per_out_ch_scaling=INTERMEDIATE_FC_PER_OUT_CH_SCALING,
                                                  bit_width=self.config.weight_bit_width,
                                                  quant_type=weight_quant_type,
                                                  stats_op=stats_op))
            in_features = out_features
            self.features.append(BatchNorm1d(num_features=in_features, eps=EPS))
            self.features.append(get_act_quant(self.config.act_bit_width, act_quant_type))
            self.features.append(Dropout(p=HIDDEN_DROPOUT))
        self.features.append(get_quant_linear(in_features=in_features,
                                   out_features=num_classes,
                                   per_out_ch_scaling=LAST_FC_PER_OUT_CH_SCALING,
                                   bit_width=self.config.weight_bit_width,
                                   quant_type=weight_quant_type,
                                   stats_op=stats_op))
        self.features.append(BatchNorm1d(num_features=num_classes, eps=EPS))

        for m in self.modules():
          if isinstance(m, QuantLinear):
            torch.nn.init.uniform_(m.weight.data, -1, 1)        
        
        if self.config.fpga:
            from .pynq_wrapper.model_wrapper import lfcWrapper 
            self.hw_plug = lfcWrapper(self.name, self.config.output_dir_path)
    
    def clip_weights(self, min_val, max_val):
        for mod in self.features:
            if isinstance(mod, QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)
    
    def normal_forward(self, x):
        for mod in self.features:
            x = mod(x)
        return x

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = 2.0 * x - torch.tensor([1.0]).type(x.type())
        if not self.config.fpga:
            x = self.normal_forward(x)
        else:
            binx = binarizeAndPack(x.numpy())
            x = self.hw_plug.forward(binx, x.shape[0])
        return x

    def creat_bin_files(self, param_dic, hw_dic, targetDir):
        import os
        os.makedirs(targetDir, exist_ok=True)
        for layer_no in range(len(param_dic)):
            weights, thresholds = param_dic["layer_" + str(layer_no)]
            numThresBits, numThresIntBits = hw_dic["layer_" + str(layer_no)][-2:]
            
            for pe in range(weights.shape[0]):
                weights[pe].astype(np.uint64).tofile(targetDir+"/"+str(layer_no)+"-"+str(pe)+"-weights.bin")
        
                if numThresIntBits is None:
                    if not np.array_equal(thresholds[pe].astype(np.int), thresholds[pe]):
                        print("WARNING: Cannot pack non-int values into binary threshold file.")
                        print("The thresholds might be processed with wrong datatype. Check BNNProcElemMem \
                            arguments numThresBits and numThresIntBits to ensure correct fractional shift.")
                else:
                    thresholds = thresholds * (1 << (numThresBits - numThresIntBits))
                thresholds[pe].astype(np.int64).tofile(targetDir+"/"+str(layer_no)+"-"+str(pe)+"-thres.bin")
        classes = [str(x) for x in range(10)]
        with open(targetDir + "/classes.txt", "w") as f:
            f.write("\n".join(classes))
        print(".bin files exported at {}".format(targetDir))

    def export(self, output_dir_path):
        param_dic, hw_dic = self.pack_params()
        self.creat_bin_files(param_dic, hw_dic, self.config.output_dir_path+self.name)

    def load_state_dict(self, *kargs, **kwargs):
        super(LFC, self).load_state_dict(*kargs, **kwargs)
        if self.config.fpga:
            import time
            start = time.time()
            self.load_params_to_fpga()
            print("Parameters loading took {} sec...".format(time.time()-start))

    def load_params_to_fpga(self):
        print("Packing and initializing network's weights and thresholds in accelerator...")
        param_dic, hw_dic = self.pack_params()
        self.hw_plug.load_parameters(param_dic, hw_dic)

    def pack_params(self):
        packed_params_dic = {}
        param_dic, hw_dic = self.get_layers_dict()
        numLayers = len(param_dic)
        import time
        import threading
        from .pynq_wrapper.params_processor import fcbn_params_to_bin_converter
        threads = []
        for i in range(numLayers):
            thread = threading.Thread(target=fcbn_params_to_bin_converter, args=(i, numLayers, param_dic["layer_" + str(i)], hw_dic["layer_" + str(i)], packed_params_dic))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        return packed_params_dic, hw_dic
    
    def get_layers_dict(self):
        import warnings
        warnings.warn("Scaling is not supported in export and fpga inference flow for now...")
        dic = {}
        hw_dic = {}
        i = 0
        for mod in self.features:
            if isinstance(mod, QuantLinear):
                dic["layer_" + str(i)] = [np.transpose(mod.weight.detach().numpy().astype(np.float64))]
                bias = mod.bias.detach().numpy().astype(np.float64) if mod.bias is not None else np.zeros(mod.weight.shape[0], dtype=np.float64)
                dic["layer_" + str(i)] += [bias]
            elif isinstance(mod, BatchNorm1d):
                dic["layer_" + str(i)] += [mod.bias.detach().numpy().astype(np.float64)]
                dic["layer_" + str(i)] += [mod.weight.detach().numpy().astype(np.float64)]
                dic["layer_" + str(i)] += [mod.running_mean.detach().numpy().astype(np.float64)]
                dic["layer_" + str(i)] += [1./np.sqrt(mod.running_var.detach().numpy().astype(np.float64)+EPS)]
                i += 1
        import json
        with open("./models/pynq_wrapper/folding_dic.json") as d:
            par_dic = json.load(d)
        par_dic = par_dic[self.name]
        for j in range(i):
            hw_dic["layer_" + str(j)] = [par_dic["pe"][j], par_dic["simd"][j]]
            hw_dic["layer_" + str(j)] += [np.array([[self.config.weight_bit_width],[par_dic["w_prec_frac"][j]]], dtype=np.uint8)]
            hw_dic["layer_" + str(j)] += [np.array([[self.config.in_bit_width if j==0 else self.config.act_bit_width],[par_dic["inp_prec_frac"][j]]], dtype=np.uint8)]
            hw_dic["layer_" + str(j)] += [np.array([[1 if j==i-1 else self.config.act_bit_width],[par_dic["out_prec_frac"][j]]], dtype=np.uint8)]
            hw_dic["layer_" + str(j)] += [par_dic["threshbits"][j], par_dic["threshIntbits"][j]]
        return dic, hw_dic