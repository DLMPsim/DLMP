# DLMP
DLMP: Agent-based simulator for distributed deep learning coordination strategies

## Citation

If you use DLMP in your research please cite:

Lopez, J., Abhari, A.
Communication Traffic Reduction in Edge-Enabled Distributed Deep Learning
Using Peer-to-Peer Coordination.
ANNSIM 2026.

## Dataset Size and Hardware Considerations

DLMP can run on a standard personal computer. However, the ability to run a specific experiment depends on the dataset size and the amount of available RAM.

Smaller datasets require less memory and can be executed on basic machines. Larger datasets require more memory and computational resources.

For example:

Dataset	Relative Size	Hardware Requirement
MNIST	Small	Runs on most laptops or desktops
CIFAR-10 / CIFAR-100	Medium	Requires more RAM and longer execution time
UA-DETRAC	Large image dataset	Recommended to run on a high-performance workstation or HPC system

Because of this difference in dataset size, the lightweight reproducible configuration in this repository uses MNIST. MNIST is small enough to run on a normal PC with limited memory.

If larger datasets are used, the following limitations may occur on a basic PC:

-insufficient RAM to load the dataset

-long training time

-reduced performance when using CPU-only execution

Users with limited hardware resources are therefore encouraged to start with the MNIST configuration, which is intended to demonstrate the simulator functionality without requiring specialized hardware.

## Requirements

-Windows 10/11 (64-bit) (can run in Linux)

-Python 3.10

-No GPU required (can run in GPU as well)

-At least 4 GB RAM (more improves runtime)

## Setup

All experiments run on CPU by default. No additional configuration is required.

python -m venv dlmp_env
dlmp_env\Scripts\activate
python -m pip install --upgrade pip
pip install torch==2.2.2 torchvision==0.17.2 mesa==2.3.2 numpy

## P2P version (EXAMPLE):

python mainMASACNN.py --dataset MNIST --processors 2 --epochs 2 --batch_size 32 --lr 0.001

## SYNC version (EXAMPLE):

python mainMASCNN.py --dataset MNIST --processors 2 --epochs 2 --batch_size 32 --lr 0.001

## Notes

- Execution runs entirely on CPU (CUDA is not required)

- MNIST is downloaded automatically on first run

- Using --processors 1 disables communication

- Runtime depends on available CPU and memory


