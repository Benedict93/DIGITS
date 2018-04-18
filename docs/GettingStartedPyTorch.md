# Getting Started with PyTorch™ in DIGITS

Table of Contents
=================
* [Enabling Support For PyTorch In DIGITS](#enabling-support-for-pytorch-in-digits)
* [Selecting PyTorch When Creating A Model In DIGITS](#selecting-tensorflow-when-creating-a-model-in-digits)
* [Defining A PyTorch Neural Network Model In DIGITS](#defining-a-pytorch-neural-network-model-in-digits)

## Enabling Support For PyTorch In DIGITS

DIGITS will automatically enable support for PyTorch if it detects PyTorch is installed in the system. This is done by a line of python code that attempts to ```import torch``` to see if it actually imports.

If DIGITS cannot enable tensorflow, a message will be printed in the console saying: ```PyTorch support is disabled```

## Selecting PyTorch When Creating A Model In DIGITS

Click on the "PyTorch" tab on the model creation page

![Select TensorFlow](images/Select_PyTorch.png)

## Defining A PyTorch Neural Network Model In DIGITS

Defining a PyTorch Neural Network Model in DIGITS is made simple and convenient because of PyTorch's [torch.nn API](http://pytorch.org/docs/master/nn.html#).

PyTorch has already provided a a container - torch.nn.module - to define the model. The tutorial as to how one can define the model can be found [here](http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)

```python
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Define layers here with functions found in torch.nn

    def forward(self, x):
        # Define forward propagation 
```

For example, this is what it looks like for [LeNet-5](http://yann.lecun.com/exdb/lenet/), a model that was created for the classification of hand written digits by Yann Lecun:

```python
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
```

The properties ```init``` and ```forward``` must be defined and the class must be called ```Net```. This is how DIGITS will interact with the python code.