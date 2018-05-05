# DiffusionMap
This package is created to conduct diffusion map with python and mpi in a user friendly way. 

Currently, several features are in plan. 
1. The only position where one really needs parallel computation is the calculation of the norm between different patterns. Therefore, it might be good to isolate that part and enable the program to do single core calculation for the other steps. 
2. Use a config.ini file to control what to do and what not to. Therefore, the users can do the calculation once and tune the visualizations without any other modifications
3. Perhaps some other kind of spectral clustering methods.
4. Solving the eigenvalues can also be done in a parallel way. However, it might not be necessary for our purpose. Therefore, at present, the program will solve the matrix within in a single node. However, the program can scan some parameters in a parallel way.