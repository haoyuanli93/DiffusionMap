# DiffusionMap

## Introduction
This package is created to conduct diffusion map with python and mpi in a user friendly way. 

This project is totally written in python. 
First, I only write python. 
Second, I don't have so much time and attention to write C++.
However, this can make the maintenance of the package challenging.
To make sure the project can be easily maintained and extended. I intend to separate functions
into different modules and with fixed interfaces. Since I have finished the first edition,
which though is not very satisfactory, I think it's time to write down a general structure 
and settle down major interfaces.

## Structure and Interface
![](/home/haoyuan/Documents/git_repos/DiffusionMap/doc/picture/structure.png)

The above image shows the general structure of the package. The data source is purely an 
interface to get access to data. The coordinator tells everyone what to do and
how to complete their jobs. The worker receive orders from the coordinator and use data source
to retrieve data to complete the job.

There are several modules in this package. The most important one is the DataSource module. 
Some of the modules do not appear in the structure diagram because they are quite independent 
or only provide some non-essential functions.

**DataSource:**
This module contains several different classes to address different data sources. Each class 
should have the following function:

    1. Divide all the data evenly into arbitrary number of batches.
    2. When is required to retrieve certain batch of patterns, returns a numpy array 
       containing the required data in the shape
                  [number of patterns, pattern shape 1, pattern shape 2]    
    3. Tell the coordinator the basic info of the datasets (how many patterns, shape, dtype).
    4. Associate a global index to each pattern in the whole dataset.
    5.  **Something compatible with dask for memory management.** (Not available in the first version)
    
**Execute:**
This module is an application of the mpi4py package together with some logic. It construct the 
coordinator and workers shown in the diagram. It told the work which kernel to use and which 
Laplacian to construct. Later, it will also include the usage of *Elemental* or *petsc4py* 
packages.

  






