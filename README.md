# brevitas-to-BNN-PYNQ

This fork demonstrates how brevitas models can be exported and executed directly on pynq board (ARM as well as fpga). It contains python scripts for hardware inference, training scripts as well as pretrained models for the LFC and CNV
used in the [BNN-PYNQ](https://github.com/Xilinx/BNN-PYNQ).

## Requirements
- Cython >= 0.29
- Numpy >= 1.18
- Pytorch >= 1.1.0
- TorchVision >= 0.4
- Brevitas (https://github.com/Xilinx/brevitas)

# Setup for pynq board
- Install the pytorch and torchvision .whl on pynqZ1/Ultra96. Compilation instructions along with the prebuilt .whls for aarch64 are available from [here](https://github.com/quetric/pynqwheels4pytorch). When compiling for arm7, set an additional flag, and the qemu to use arm7. Follow rest of the given compilation instructions as it is.
 ```bash
export NO_DISTRIBUTED=1
echo ':qemu-arm:M::\x7fELF\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x28\x00:\xff\xff\xff\xff\xff\xff\xff\x00\xff\xff\xff\xff\xff\xff\xff\xff\xfe\xff\xff\xff:/usr/bin/qemu-arm-static:' > /proc/sys/fs/binfmt_misc/register
 ```
- Copy the .whl files on the pynq board and install:

 ```bash
sudo pip3 install *.whl
 ```

- Install Brevitas and pre-requisites:
 ```bash
 sudo pip3 install packaging Cython==0.29 numpy==1.18
 git clone https://github.com/Xilinx/brevitas.git
 cd brevitas
 sudo pip3 install .
 ```

- On the pynq board, with Brevitas installed, clone the this repo and build an .so file for fast parameter loading:
 ```bash
 git clone https://github.com/ussamazahid96/brevitas_cnv_lfc.git
 cd brevitas_cnv_lfc/training_scripts/models/pynq_wrapper/
 ./gen_layer_loader.sh
 ```

## Evaluate on fpga

The included bitstreams are taken as it is from [BNN-PYNQ](https://github.com/Xilinx/BNN-PYNQ). From within *training_scripts* folder:
 ```bash
sudo python3 main.py --network <LFC/CNV> --dataset <MNIST/CIFAR10> --weight_bit_width <1/2> --act_bit_width <1/2> --in_bit_width <1/8> --resume ../pretrained_models/<selected model>/checkpoints/best.tar --evaluate --fpga
 ```

## Evaluate (on CPU/ARM)

Similarly to run inference on PC or ARM processor of pynq board, run the same command with the additional `--gpus None` flag:
 ```bash
python3 main.py --network <LFC/CNV> --dataset <MNIST/CIFAR10> --weight_bit_width <1/2> --act_bit_width <1/2> --in_bit_width <1/8> --resume ../pretrained_models/<selected model>/checkpoints/best.tar --evaluate --gpus None
 ```

## Training on GPU

Install Pytorch, Torchvision and Brevitas on PC. An experiments folder at */path/to/experiments* must exist before launching the training:
 ```bash
python3 main.py --network <LFC/CNV> --dataset <MNIST/CIFAR10> --weight_bit_width <1/2> --act_bit_width <1/2> --in_bit_width <1/8> --experiments /path/to/experiments
 ```

## Export .bin files to use in BNN-PYNQ

.bin files to use in BNN-PYNQ for bitstream generation can be exported as:
```bash
sudo python3 main.py --network <LFC/CNV> --dataset <MNIST/CIFAR10> --weight_bit_width <1/2> --act_bit_width <1/2> --in_bit_width <1/8> --resume ../pretrained_models/<selected model>/checkpoints/best.tar --export
 ```