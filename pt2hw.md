The following changes to applied for accurate pytorch -> hardware flow:

## During Training:

1. During training quantize the input in Q1.7 format as the hardware is using during the inference. Check 
[here](https://github.com/ussamazahid96/BNN-PYNQ/blob/fffddb17c3ce865487444aab2448d50d2cfd5f21/bnn/src/training/Pytorch/lenet_w8a8.py#L77)
how to do that.

2. When training the network with Pytorch (no matter what precision) always use MaxPool after the activation layer because the hardware
is applying max-pool on the activations i.e. after the thresholding operation (it is strange but PyTorch gives many negative thresholds
which on hardware reflects as a min pool operation whereas Theano never gives negative thresholds in Conv layers so its always a 
max pool even if you change the order on hardware).

3. The last layer on hardware outputs the accumulation values as it is and does not use thresholding, But for smooth training, 
we require a batch-norm layer after the last FC layer. If you check the Theano training scripts it uses a special batch-norm 
layer at the end. Check [here](https://github.com/Xilinx/BNN-PYNQ/blob/eb19cb5ccc2e10f066525fdbce257620ef685f29/bnn/src/training/cnv.py#L237)
which is from the original BNN-PYNQ repo. This special batch-norm layer has one value each i.e. for mean, 
invstd, beta, gamma rather than ten. This batch-norm layer is not available in Pytorch, but I have written one and it is available
[here](https://github.com/ussamazahid96/brevitas_cnv_lfc/blob/master/training_scripts/models/layernorm.py). 
You have to use this batch-norm layer at the end rather than the one having ten values for each parameter.

 4. When comparing the accuracy of the hardware you should always compare it with the CPU accuracy of the Pytorch model.
 
 ## During Export:
 
 1. The invstd value of every batch-norm layer in Theano already has the eps (1e-4) included in it, But in Pytorch you have to include it 
 yourself while exporting Check [here](https://github.com/ussamazahid96/BNN-PYNQ/blob/fffddb17c3ce865487444aab2448d50d2cfd5f21/bnn/src/training/Pytorch/lenet_w8a8.py#L112).
 
 2. In the [finnthesizer.py](https://github.com/Xilinx/BNN-PYNQ/blob/master/bnn/src/training/finnthesizer.py) replace all the 
 np.ceil operations with np.floor.
 
 ## During Inference:
 
 1. The most important difference in Theano and Pytorch is that if in the output predictions, more than one class has the same 
 highest score (which happens almost always that you end up with the same highest score for two classes). Now Theano assigns the 
 class with the lower index whereas Pytorch assigns the class with the higher index. And the host code on BNN-PYNQ is also following 
 the Theano. So you have to change all the ">" with ">=" e.g. [here](https://github.com/ussamazahid96/BNN-PYNQ/blob/fffddb17c3ce865487444aab2448d50d2cfd5f21/bnn/src/library/host/foldedmv-offload.cpp#L157)
 and similarily for LFC networks all the "0" to "9" e.g. [here](https://github.com/ussamazahid96/BNN-PYNQ/blob/fffddb17c3ce865487444aab2448d50d2cfd5f21/bnn/src/library/host/foldedmv-offload.cpp#L157).
 
 
 
