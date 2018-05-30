"""
This scripts add all h5 files under a specified folder to a specified txt file.
That txt file is intended to be used as the instruction for the program to find
the data set to process.
"""

import os
import argparse

# Set up the parser
parser = argparse.ArgumentParser()
parser.add_argument('source_folder', type=int, help="batch number")
parser.add_argument('file_list', type=str, help="The output txt file for this list.")

# Parse
args = parser.parse_args()
source_folder = args.source_folder
file_list = args.file_list

# Search the folder
search_result = os.listdir(source_folder)

# Construct the text file content
content = []
for l in search_result:
    if l[-3:] == ".h5":
        absolute_path = os.path.abspath(source_folder + "/" + l)
        content.append(str(absolute_path))
        print(absolute_path)

# Sort the list according to lexicographical order
content.sort(key=str.lower)
for line in content:
    print(line)

# Write content to the txt file
"""
Notice that these two loops can be composed into one.
But since it does not take very long time, and should not,
therefore, I keep them separated.
"""
with open(file_list, "w") as txtfile:
    for l in content:
        txtfile.write("File:")
        txtfile.write(l + "\n")

# Finish can tell the user to double check the list.
print("The absolute paths of all h5 files in folder")
print(source_folder)
print("have been added to txt file {}. Please check the file".format(file_list))
print("to see if it has put every file you want in this list.")
