"""
This script create a project folder to execute the diffusion map.
"""
import os
import shutil

for l in range(1000):

    # Find the first available name and create the project folder
    project_dir = './proj_{:0>3d}'.format(l)
    if os.path.exists(project_dir):
        continue

    # Create a directory with the first available name
    os.makedirs(project_dir)

    # Create standard folder in this project folder
    os.makedirs(project_dir + '/input')
    os.makedirs(project_dir + '/output')
    os.makedirs(project_dir + '/src')

    # Create useful files for the user
    with open(project_dir + '/input/file_list.txt', 'w') as txtfile:
        txtfile.write("##################################################################\n")
        txtfile.write("# Dear users:\n")
        txtfile.write("# This is a file aims to help you to specify which dataset to process.\n")
        txtfile.write("# This txt file, or any txt file obeying the following roles can\n")
        txtfile.write("# be understood by the program.\n")
        txtfile.write("# 1. The program ignore any lines that is empty, starts with # or blank space. \n")
        txtfile.write("# 2. The program can not handle addresses containing any blank spaces.\n")
        txtfile.write("#    Please pay attention to this.\n")
        txtfile.write("# 3. The h5py file address, preferably the absolute address, should \n")
        txtfile.write("#    follow \"File:the_address_no_blank_space\"\n")
        txtfile.write("# 4. The dataset in the hdf5 file to process is listed right below\n")
        txtfile.write("#    the corresponding hdf5 file line following \"Dataset:\". Each \n")
        txtfile.write("#    dataset should occupy a whole line.\n")
        txtfile.write("# 5. If one does not specify which dataset to process, the default\n")
        txtfile.write("#    action is to process all the datasets contained in this hdf5 file.\n")
        txtfile.write("# 6. The program detects if there are any duplicated files. Please\n")
        txtfile.write("#    make sure that all the h5py files are unique.\n")
        txtfile.write("# 7. The program does not detect if the datasets are unique, therefore\n")
        txtfile.write("#    please pay extra caution to this since this can incur errors\n")
        txtfile.write("#    that are extremely difficult to trace down.\n")
        txtfile.write("# \n")
        txtfile.write("# Best wishes.\n")
        txtfile.write("# Haoyuan Li\n")
        txtfile.write("# 7/17/2018 \n")
        txtfile.write("# \n")
        txtfile.write("#-----------------------------------------------------------------\n")
        txtfile.write("#             *This is an example*\n")
        txtfile.write("#-----------------------------------------------------------------\n")
        txtfile.write("#File:/reg/d/psdm/amo/amo86615/res/haoyuan/singles.h5\n")
        txtfile.write("#Dataset:patterns_one\n")
        txtfile.write("#Dataset:patterns_two\n")
        txtfile.write("#File:/reg/d/psdm/amo/amo86615/res/haoyuan/multiples.py\n")
        txtfile.write("##################################################################\n")

    # Copy the Config.py file
    shutil.copyfile(src='./asset/Config.py', dst=project_dir + '/src/Config.py')

    # Copy the WeightMat.py file
    shutil.copyfile(src='./WeightMat.py', dst=project_dir + '/src/WeightMat.py')

    # Copy the EigensSlepc.py file
    shutil.copyfile(src='./EigensSlepc.py', dst=project_dir + '/src/EigensSlepc.py')

    # Only create one new project folder
    break

print("The new project is located at {}".format(project_dir))
