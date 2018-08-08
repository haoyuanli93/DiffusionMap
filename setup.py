from setuptools import setup, find_packages

setup(
    name='pDiffusionMap',
    version=0.2,
    description=(
        'This package aims to do Diffusion Map in a parallel way. '),
    author='Haoyuan Li',
    author_email='hyli16@stanford.edu',
    maintainer='Haoyuan Li',
    maintainer_email='hyli16@stanford.edu',
    license='BSD License',
    packages=["abbr", "DataSource", "Graph", "util", "visutil"],
    platforms=["Linux"],
    url='https://github.com/haoyuanli93/DiffusionMap',
    classifiers=[
        'Development Status :: 0.2',
        'Operating System :: Linux',
        'Intended Audience :: Single Particle Imaging scientists',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries'
    ],
)
