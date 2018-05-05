"""
This parser reads a config.ini file to generally control all the calculations,
io and visualizations. The basic idea is the following:

This parser check a config.ini file, which is a txt file, twice.

The first time this parser reads through the config.ini file, it loads every thing in the configuration file
into the memory as a dictionary. The keys of the dictionary indicate what the program is going to do.
In addition to the keys for the concrete actions, there is one more entry in the dictionary. This energy is
a list, this list contains all the keys of the dictionary and indicates what is the sequence of actions.

If a sequence is provided, then the calculation is conducted with the specified sequence, otherwise, the sequence
the default sequence.
"""

"""
Step one. Read the file and save them into different 

"""



