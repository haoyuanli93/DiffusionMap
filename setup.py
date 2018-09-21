from setuptools import setup

setup(
    name='pDiffusionMap',
    version='0.2.5',
    description=(
        'This package aims to do Diffusion Map in a parallel way. '),
    long_description=("This package calculate the diffusion map in a parallel manner.\n "
                      "Please check the github repo at https://github.com/haoyuanli93/DiffusionMap \n"
                      "for more detailed description."),
    author='Haoyuan Li',
    author_email='hyli16@stanford.edu',
    maintainer='Haoyuan Li',
    maintainer_email='hyli16@stanford.edu',
    license='BSD License',
    packages=["pDiffusionMap",],
    install_requires=['numpy',
                      'holoviews',
                      'datashader',
                      'matplotlib',
                      'pandas',
                      'h5py',
                      'petsc4py',
                      'slepc4py',
                      'mpi4py',
                      'ipyvolume'],
    platforms=["Linux"],
    url='https://github.com/haoyuanli93/DiffusionMap'
)
