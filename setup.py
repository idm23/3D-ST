"""Setup script for 3D-ST
"""
# ======== standard imports ========
# ==================================

# ======= third party imports ======
# ==================================

# ========= program imports ========
import setuptools
# ==================================

with open("README.md", 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='st3d',
    version='0.9.1',
    description=long_description,
    author='Ian Mackey',
    author_email='idm@ianmackey.net',
    packages=setuptools.find_packages(include=['st3d', 'st3d*']),
    python_requires='<3.12',
    install_requires=[
        'torch>=2.1.0', # There are issues with this. Torch is required but installation isn't so simple.
        'numpy>=1.26.0',
        'matplotlib',
        'tqdm',
        'numba'
    ],
)