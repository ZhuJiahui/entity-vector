# -*- coding: utf-8 -*-

import numpy as np
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

setup(
    name='entity-vector',
    version='0.0.2',
    description='Joint embedding of words and Wikipedia entities',
    author='Studio Ousia',
    author_email='ikuya@ousia.jp',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'entity-vector=entity_vector:cli',
        ]
    },
    ext_modules=cythonize([
        Extension(
            '*', ['entity_vector/*.pyx'],
            include_dirs=[np.get_include()],
        ),
        Extension(
            '*', ['entity_vector/utils/tokenizer/*.pyx'],
        )
    ]),
)
