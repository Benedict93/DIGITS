# Installing PyTorch
PyTorch on DIGITS is still a work in progress

Installation for Ubuntu and Mac can be found at [here](http://pytorch.org/)

Recommended Package manager to install PyTorch: Anaconda

## Requirements

DIGITS is current targeting PyTorch 0.3.1

TensorFlow for DIGITS requires one or more NVIDIA GPUs with CUDA Compute Capbility of 3.0 or higher. See [the official GPU support list](https://developer.nvidia.com/cuda-gpus) to see if your GPU supports it.

Along with that requirement, the following should be installed

* One or more NVIDIA GPUs ([details](InstallCuda.md#gpu))
* An NVIDIA driver ([details and installation instructions](InstallCuda.md#driver))
* A CUDA toolkit ([details and installation instructions](InstallCuda.md#cuda-toolkit))
	- Current CUDA supported: version 8
* cuDNN 5.1 ([download page](https://developer.nvidia.com/cudnn))

Ensure that you are on the latest pip and numpy packages


## Installation

These instructions are based on [the official PyTorch instructions](http://pytorch.org/)

PyTorch can be installed via 3 different methods: Anaconda package manager, pip and source

To install via anaconda, use the following command:

```
conda install pytorch torchvision -c pytorch
```

To install via pip, use the following command:

```
# For python 2.7
pip install torch torchvision

# For python 3.5 and 3.6
pip3 install torch torchvision
```

To install via source, go to the following [link](https://github.com/pytorch/pytorch#from-source)

PyTorch should then install effortlessly and pull in all its required dependencies

## Getting Started With TensorFlow In DIGITS

Follow [these instructions](GettingStartedPyTorch.md) for information on getting started with PyTorch in DIGITS
