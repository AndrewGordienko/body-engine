from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

os.environ["CC"] = "/usr/bin/clang"
os.environ["CXX"] = "/usr/bin/clang++"

# Directory paths
MUJOCO_HEADERS_DIR = '/usr/local/include/mujoco'
GLFW_LIB_DIR = '/opt/homebrew/opt/glfw/lib'
EIGEN_INCLUDE_DIR = '/opt/homebrew/opt/eigen/include/eigen3'
TINYXML2_LIB_DIR = '/opt/homebrew/lib'
TINYXML2_INCLUDE_DIR = '/opt/homebrew/include'

extensions = [
    Extension(
        name="myenv",
        sources=[
            'src/GLFWWindowManager.cpp',
            'src/main.cpp'
        ],
        include_dirs=[
            np.get_include(),
            MUJOCO_HEADERS_DIR,
            EIGEN_INCLUDE_DIR,
            '/usr/local/include',
            TINYXML2_INCLUDE_DIR,
            'include'
        ],
        library_dirs=[
            GLFW_LIB_DIR,
            TINYXML2_LIB_DIR,
            '/opt/homebrew/opt/llvm/lib',
            '/opt/homebrew/Cellar/llvm/17.0.6_1/lib/c++'
        ],
        libraries=[
            'glfw',
            'tinyxml2',
            'c++'
        ],
        extra_compile_args=[
            '-std=c++17',
            '-fcolor-diagnostics',
            '-fansi-escape-codes',
            '-g',  # Add this line for debugging information
        ],
        extra_link_args=[
            '-F/Applications',
            '-framework', 'mujoco',
            '-Wl,-rpath,/Applications/MuJoCo.app/Contents/Frameworks',
            '-L/opt/homebrew/opt/llvm/lib',
            '-L/opt/homebrew/Cellar/llvm/17.0.6_1/lib/c++',
            '-lc++'
        ],
        language='c++',
    )
]

setup(
    ext_modules=cythonize(extensions, language_level=3),
    include_dirs=[np.get_include()],
)
