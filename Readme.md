
# Implementation for verifiable Data Quality in Federated Learning

This is the implementation of our idea for an extension for verifiable data quality in federated learning.

As an example, we introduce four different quality dimensions on the [MNIST dataset](https://yann.lecun.com/exdb/mnist/).
For multiple trainers with different quality data, each dimension is checked, and a zero-knowledge proof is created.

## Requirements

The Proof generation is done with the [Zokrates](https://zokrates.github.io/) toolbox and allows for different proving schemes, the default is Nova.
The Setup in Zokrates does not work on all operating systems; check the [Examples](https://zokrates.github.io/toolbox/) first.

Additional requirements include any Python packages in Local_Trainers.py

## Start
Either start the process directly with Local_Trainers.py or use run.sh for multiple runs using SLURM.
# Contents

## Data
The [MNIST dataset](https://yann.lecun.com/exdb/mnist/) used in our Implementation

## FederatedLearning
The Main implementation of the verifiable data quality phase for federated learning.

## MNISTeval
The different evaluations for different data quality dimensions on a given convolutional neural network.

## Zokratestest
Some simple Zokrates and MNIST functionality tests.
