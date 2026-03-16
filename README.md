# DLMP
DLMP: Agent-based simulator for distributed deep learning coordination strategies

## Citation

If you use DLMP in your research please cite:

Lopez, J., Abhari, A.
Communication Traffic Reduction in Edge-Enabled Distributed Deep Learning
Using Peer-to-Peer Coordination.
ANNSIM 2026.

Dataset Size and Hardware Considerations

DLMP can run on a standard personal computer. However, the ability to run a specific experiment depends on the dataset size and the amount of available system memory (RAM).

Smaller datasets require less memory and can be executed on basic machines. Larger datasets require more memory and computational resources.

For example:

Dataset	Relative Size	Hardware Requirement
MNIST	Small	Runs on most laptops or desktops
CIFAR-10 / CIFAR-100	Medium	Requires more RAM and longer execution time
UA-DETRAC	Large image dataset	Recommended to run on a high-performance workstation or HPC system

Because of this difference in dataset size, the lightweight reproducible configuration in this repository uses MNIST. MNIST is small enough to run on a normal PC with limited memory.

If larger datasets are used, the following limitations may occur on a basic PC:

insufficient RAM to load the dataset

long training time

reduced performance when using CPU-only execution

Users with limited hardware resources are therefore encouraged to start with the MNIST configuration, which is intended to demonstrate the simulator functionality without requiring specialized hardware.
