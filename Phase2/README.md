# Phase 2: CIFAR-10 Classification

This repository contains the code for training and testing a neural network on the CIFAR-10 dataset using PyTorch. The network can be one of several predefined architectures, including simple CNNs, ResNet, ResNeXt, and DenseNet.

## Table of Contents

- [Installation](#installation)
- [Usage](#setup)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Prerequisites

To run the code in this repository, you need to have the following installed:

- PyTorch
- torchvision
- scikit-image
- numpy
- matplotlib
- argparse
- tensorboard
- termcolor

You can install these packages using pip:

```bash
pip install torch torchvision scikit-image numpy matplotlib argparse tensorboard termcolor
```

## Usage

* Clone this repository and navigate to the `Phase2/Code` directory:

```bash
git clone <repository-url>
cd Phase2/Code
```

* Training: To train the model, run the 'Train.py' script:
```bash
python Train.py --CheckPointPath 'Phase2/Code/Checkpoints/' --NumEpochs 50 --DivTrain 1 --MiniBatchSize 64 --LoadCheckPoint 0 --LogsPath 'Phase2/Code/Logs/'
```
... __Arguments__
..* --CheckPointPath: Path to save checkpoints (default: Phase2/Code/Checkpoints/)
..* --NumEpochs: Number of epochs to train (default: 50)
..* --DivTrain: Factor to reduce train data by per epoch (default: 1)
..* --MiniBatchSize: Size of the mini-batch to use (default: 64)
..* --LoadCheckPoint: Load model from latest checkpoint (default: 0)
..* --LogsPath: Path to save logs for TensorBoard (default: Phase2/Code/Logs/)

* Testing: To test the model, run the 'Test.py' script:
```bash
python Test.py --ModelPath 'Phase2/Code/Checkpoints/0model.ckpt' --LabelsPath 'Phase2/Code/TxtFiles/LabelsTest.txt'
```
... __Arguments__
..* --ModelPath: Path to load the latest model from (default: Phase2/Code/Checkpoints/DenseNetWS.ckpt)
..* --LabelsPath: Path of labels file (default: Phase2/Code/TxtFiles/LabelsTest.txt)

## Results
The training script saves the model checkpoints and TensorBoard logs in the specified directories. After training, the script generates plots for training loss and accuracy over epochs and saves them in the 'Phase2/Code/Checkpoints/' directory.

The testing script outputs the confusion matrix and accuracy of the model on the test set. The confusion matrix is saved as an image in the 'Phase2/Code/Plot/' directory.