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

import torch
from torch.nn import Module, ModuleList, BatchNorm2d, MaxPool2d, BatchNorm1d, Sequential

from .layernorm import LayerNorm
from .common import get_quant_conv2d, get_quant_linear, get_act_quant, get_quant_type, get_stats_op

from brevitas.core.scaling import ScalingImplType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.nn import QuantConv2d, QuantHardTanh, QuantLinear

from .pynq_wrapper.input_processor import quantizeAndPack

# QuantConv2d configuration
CNV_OUT_CH_POOL = [(64, False), (64, True), (128, False), (128, True), (256, False), (256, False)]

# Intermediate QuantLinear configuration
INTERMEDIATE_FC_PER_OUT_CH_SCALING = False
INTERMEDIATE_FC_FEATURES = [(256, 512), (512, 512)]

# Last QuantLinear configuration
LAST_FC_IN_FEATURES = 512
LAST_FC_PER_OUT_CH_SCALING = False

# MaxPool2d configuration
POOL_SIZE = 2
EPS=1e-4

class CNV(Module):

    def __init__(self, config, num_classes=10, in_ch=3):
        super(CNV, self).__init__()

        self.config = config
        self.name = "cnvW{}A{}".format(self.config.weight_bit_width, self.config.act_bit_width)
        weight_quant_type = get_quant_type(self.config.weight_bit_width)
        act_quant_type = get_quant_type(self.config.act_bit_width)
        in_quant_type = get_quant_type(self.config.in_bit_width)
        stats_op = get_stats_op(weight_quant_type)
        max_in_val = 1-2**(-7) # for Q1.7 input format
        self.conv_features = ModuleList()
        self.linear_features = ModuleList()

        self.conv_features.append(QuantHardTanh(bit_width=self.config.in_bit_width,
                                                quant_type=in_quant_type,
                                                max_val=max_in_val,
                                                restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                                scaling_impl_type=ScalingImplType.CONST))

        for out_ch, is_pool_enabled in CNV_OUT_CH_POOL:
            self.conv_features.append(get_quant_conv2d(in_ch=in_ch,
                                                       out_ch=out_ch,
                                                       bit_width=self.config.weight_bit_width,
                                                       quant_type=weight_quant_type,
                                                       stats_op=stats_op))
            in_ch = out_ch
            self.conv_features.append(BatchNorm2d(in_ch, eps=EPS))
            self.conv_features.append(get_act_quant(self.config.act_bit_width, act_quant_type))
            if is_pool_enabled:
                self.conv_features.append(MaxPool2d(kernel_size=2))

        for in_features, out_features in INTERMEDIATE_FC_FEATURES:
            self.linear_features.append(get_quant_linear(in_features=in_features,
                                                         out_features=out_features,
                                                         per_out_ch_scaling=INTERMEDIATE_FC_PER_OUT_CH_SCALING,
                                                         bit_width=self.config.weight_bit_width,
                                                         quant_type=weight_quant_type,
                                                         stats_op=stats_op))
            self.linear_features.append(BatchNorm1d(out_features, eps=EPS))
            self.linear_features.append(get_act_quant(self.config.act_bit_width, act_quant_type))
        
        self.linear_features.append(get_quant_linear(in_features=LAST_FC_IN_FEATURES,
                                   out_features=num_classes,
                                   per_out_ch_scaling=LAST_FC_PER_OUT_CH_SCALING,
                                   bit_width=self.config.weight_bit_width,
                                   quant_type=weight_quant_type,
                                   stats_op=stats_op))
        self.linear_features.append(LayerNorm())
        
        for m in self.modules():
          if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
            torch.nn.init.uniform_(m.weight.data, -1, 1)        

        if self.config.fpga:
            from .pynq_wrapper.model_wrapper import cnvWrapper 
            self.hw_plug = cnvWrapper(self.name, self.config.output_dir_path)

    def clip_weights(self, min_val, max_val):
        for mod in self.conv_features:
            if isinstance(mod, QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.linear_features:
            if isinstance(mod, QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)

    def normal_forward(self, x):
        for mod in self.conv_features:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.linear_features:
            x = mod(x)
        return x
    
    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0]).type(x.type())
        if not self.config.fpga:
            x = self.normal_forward(x)
        else:
            quantx = quantizeAndPack(x.numpy())
            x = self.hw_plug.forward(quantx, x.shape[0])  
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
                thresholds[pe].astype(np.int64).tofile(targetDir+"/"+str(layer_no)+"-"+str(pe)+"-thres.bin")
        classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
        with open(targetDir + "/classes.txt", "w") as f:
            f.write("\n".join(classes))
        print(".bin files exported at {}".format(targetDir))

    def export(self, output_dir_path):
        param_dic, hw_dic = self.pack_params()
        self.creat_bin_files(param_dic, hw_dic, self.config.output_dir_path+self.name)

    def load_state_dict(self, *kargs, **kwargs):
        super(CNV, self).load_state_dict(*kargs, **kwargs)
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
        import threading
        from .pynq_wrapper.params_processor import model_params_to_bin_converter          
        threads = []
        for i in range(numLayers):
            thread = threading.Thread(target=model_params_to_bin_converter, args=(i, self.num_conv_layers, numLayers, param_dic["layer_" + str(i)], hw_dic["layer_" + str(i)], packed_params_dic))
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
        first_fc = True
        for mod in [x for x in self.conv_features] + [y for y in self.linear_features]:
            if isinstance(mod, QuantConv2d):
                weights = mod.weight.detach().numpy().astype(np.float64)
                numInterleaveChannels = weights.shape[0]
                dic["layer_" + str(i)] = [weights]
                bias = mod.bias.detach().numpy().astype(np.float64) if mod.bias is not None else np.zeros(mod.weight.shape[0], dtype=np.float64)
                dic["layer_" + str(i)] += [bias]
            elif isinstance(mod, QuantLinear):
                weights = mod.weight.detach().numpy().astype(np.float64)
                if first_fc:
                    weights = weights.reshape(weights.shape[0], numInterleaveChannels, -1)
                    weights = weights.swapaxes(1,-1).reshape(weights.shape[0], -1)
                    self.num_conv_layers = i
                    first_fc = False
                dic["layer_" + str(i)] = [np.transpose(weights)]
                bias = mod.bias.detach().numpy().astype(np.float64) if mod.bias is not None else np.zeros(mod.weight.shape[0], dtype=np.float64)
                dic["layer_" + str(i)] += [bias]
            elif isinstance(mod, (BatchNorm2d, BatchNorm1d, LayerNorm)):
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
            hw_dic["layer_" + str(j)] += [np.array([[1 if j==0 else self.config.act_bit_width],[par_dic["inp_prec_frac"][j]]], dtype=np.uint8)]
            hw_dic["layer_" + str(j)] += [np.array([[0 if j==i-1 else self.config.act_bit_width],[par_dic["out_prec_frac"][j]]], dtype=np.uint8)]
            hw_dic["layer_" + str(j)] += [par_dic["threshbits"][j], par_dic["threshIntbits"][j]]
        return dic, hw_dic
