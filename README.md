# SHREC 2025: Partial Retrieval Benchmark - Replication Archive

This is the code repository for the evaluation strategy described in the paper:

(TODO)

[Link to paper](https://todo)

### Getting started

Clone the repository using the --recursive flag:
```
git clone https://github.com/bartvbl/SHREC2025-Partial-Retrieval-Benchmark-Replication-Archive --recursive
```
You should subsequently run the python script to install dependencies, compile the project, and download precompiled cache files. You can use the following command to do so:
```
python3 run_and_replicate.py
```
Refer to the included PDF file for information about replicating the results produced for the paper.

### System requirements
* 32GB of RAM (64GB is highly recommended)
* The project has been tested on Linux Mint 22.1.
* The CUDA SDK must be installed, but no GPU is required to run the benchmark, except for methods that utilise machine learning
* Python 3.10 or above

There's nothing that should prevent the project to be compiled on Windows, but it has not been tested.

While many of the experiments and filters will run on a system with 32GB of RAM, you will likely need to apply a number of thread limiters in order to reduce memory requirements. Make in any case sure to have enough swap space available, and a healthy dose of patience when using thread limiters.

### Abstract

Partial retrieval is a long-standing problem in the 3D Object Retrieval community. Its main difficulties arise from how to define 3D local descriptors in a way that makes them effective for partial retrieval and robust to common real-world issues, such as occlusion, noise, or clutter, when dealing with 3D data. This SHREC track is based on the newly proposed ShapeBench benchmark to evaluate the matching performance of local descriptors. We propose an experiment consisting of three increasing levels of difficulty, where we combine different filters to simulate real-world issues related to the partial retrieval task. Our main findings show that classic 3D local descriptors like Spin Image are robust to several of the tested filters (and their combinations), but more recent learned local descriptors like GeDI can be competitive for some specific filters. Finally, no 3D local descriptor was able to successfully handle the hardest level of difficulty.

### Citation

```
(TODO)
```

### Troubleshooting
To ensure compatibility with the default LibEigen3 installation from `apt`, please modify line `27` of [CMakeLists](https://github.com/bartvbl/ShapeBench-Replication-Archive/blob/main/CMakeLists.txt) to
```
find_package(Eigen3 3.3.0 REQUIRED)
```
