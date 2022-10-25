#
# GeomProc: geometry processing library in python + numpy
#
# Copyright (c) 2008-2021 Oliver van Kaick <ovankaic@gmail.com>
# under the MIT License.
# For all the parts of the code where authorship is not explicitly
# mentioned.
#
# Marching cubes code based on C++ code written by Matthew Fisher (2014)
# (https://graphics.stanford.edu/~mdfisher/MarchingCubes.html)
# License: unknown. Highly similar C++ code exists with
# Copyright (C) 2002 by Computer Graphics Group, RWTH Aachen
# under the GPL license.
#
# See file LICENSE.txt for details on copyright licenses.
#
"""This module contains marching cubes related functions of the GeomProc
geometry processing library.
"""


import numpy as np

from .mesh import *


# Marching cubes
#
# The following code is based on C++ code made available by Matthew Fisher
# https://graphics.stanford.edu/~mdfisher/index.html

# Marching cubes table data
#
# Table that maps each cube case (based on the signs of the implicit
# function computed at the corners of the cube) to the indices of the
# edges where we should create a vertex
edge_table = [ \
    0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, \
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00, \
    0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, \
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, \
    0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c, \
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30, \
    0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac, \
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, \
    0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c, \
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60, \
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc, \
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0, \
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c, \
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950, \
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc , \
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, \
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, \
    0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0, \
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, \
    0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650, \
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, \
    0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0, \
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, \
    0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460, \
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, \
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0, \
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, \
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230, \
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, \
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190, \
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, \
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0] 

# Table that stores the triangulation for each cube case
tri_table = [ \
    [], \
    [0, 8, 3], \
    [0, 1, 9], \
    [1, 8, 3, 9, 8, 1], \
    [1, 2, 10], \
    [0, 8, 3, 1, 2, 10], \
    [9, 2, 10, 0, 2, 9], \
    [2, 8, 3, 2, 10, 8, 10, 9, 8], \
    [3, 11, 2], \
    [0, 11, 2, 8, 11, 0], \
    [1, 9, 0, 2, 3, 11], \
    [1, 11, 2, 1, 9, 11, 9, 8, 11], \
    [3, 10, 1, 11, 10, 3], \
    [0, 10, 1, 0, 8, 10, 8, 11, 10], \
    [3, 9, 0, 3, 11, 9, 11, 10, 9], \
    [9, 8, 10, 10, 8, 11], \
    [4, 7, 8], \
    [4, 3, 0, 7, 3, 4], \
    [0, 1, 9, 8, 4, 7], \
    [4, 1, 9, 4, 7, 1, 7, 3, 1], \
    [1, 2, 10, 8, 4, 7], \
    [3, 4, 7, 3, 0, 4, 1, 2, 10], \
    [9, 2, 10, 9, 0, 2, 8, 4, 7], \
    [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4], \
    [8, 4, 7, 3, 11, 2], \
    [11, 4, 7, 11, 2, 4, 2, 0, 4], \
    [9, 0, 1, 8, 4, 7, 2, 3, 11], \
    [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1], \
    [3, 10, 1, 3, 11, 10, 7, 8, 4], \
    [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4], \
    [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3], \
    [4, 7, 11, 4, 11, 9, 9, 11, 10], \
    [9, 5, 4], \
    [9, 5, 4, 0, 8, 3], \
    [0, 5, 4, 1, 5, 0], \
    [8, 5, 4, 8, 3, 5, 3, 1, 5], \
    [1, 2, 10, 9, 5, 4], \
    [3, 0, 8, 1, 2, 10, 4, 9, 5], \
    [5, 2, 10, 5, 4, 2, 4, 0, 2], \
    [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8], \
    [9, 5, 4, 2, 3, 11], \
    [0, 11, 2, 0, 8, 11, 4, 9, 5], \
    [0, 5, 4, 0, 1, 5, 2, 3, 11], \
    [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5], \
    [10, 3, 11, 10, 1, 3, 9, 5, 4], \
    [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10], \
    [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3], \
    [5, 4, 8, 5, 8, 10, 10, 8, 11], \
    [9, 7, 8, 5, 7, 9], \
    [9, 3, 0, 9, 5, 3, 5, 7, 3], \
    [0, 7, 8, 0, 1, 7, 1, 5, 7], \
    [1, 5, 3, 3, 5, 7], \
    [9, 7, 8, 9, 5, 7, 10, 1, 2], \
    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3], \
    [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2], \
    [2, 10, 5, 2, 5, 3, 3, 5, 7], \
    [7, 9, 5, 7, 8, 9, 3, 11, 2], \
    [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11], \
    [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7], \
    [11, 2, 1, 11, 1, 7, 7, 1, 5], \
    [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11], \
    [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0], \
    [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0], \
    [11, 10, 5, 7, 11, 5], \
    [10, 6, 5], \
    [0, 8, 3, 5, 10, 6], \
    [9, 0, 1, 5, 10, 6], \
    [1, 8, 3, 1, 9, 8, 5, 10, 6], \
    [1, 6, 5, 2, 6, 1], \
    [1, 6, 5, 1, 2, 6, 3, 0, 8], \
    [9, 6, 5, 9, 0, 6, 0, 2, 6], \
    [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8], \
    [2, 3, 11, 10, 6, 5], \
    [11, 0, 8, 11, 2, 0, 10, 6, 5], \
    [0, 1, 9, 2, 3, 11, 5, 10, 6], \
    [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11], \
    [6, 3, 11, 6, 5, 3, 5, 1, 3], \
    [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6], \
    [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9], \
    [6, 5, 9, 6, 9, 11, 11, 9, 8], \
    [5, 10, 6, 4, 7, 8], \
    [4, 3, 0, 4, 7, 3, 6, 5, 10], \
    [1, 9, 0, 5, 10, 6, 8, 4, 7], \
    [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4], \
    [6, 1, 2, 6, 5, 1, 4, 7, 8], \
    [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7], \
    [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6], \
    [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9], \
    [3, 11, 2, 7, 8, 4, 10, 6, 5], \
    [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11], \
    [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6], \
    [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6], \
    [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6], \
    [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11], \
    [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7], \
    [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9], \
    [10, 4, 9, 6, 4, 10], \
    [4, 10, 6, 4, 9, 10, 0, 8, 3], \
    [10, 0, 1, 10, 6, 0, 6, 4, 0], \
    [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10], \
    [1, 4, 9, 1, 2, 4, 2, 6, 4], \
    [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4], \
    [0, 2, 4, 4, 2, 6], \
    [8, 3, 2, 8, 2, 4, 4, 2, 6], \
    [10, 4, 9, 10, 6, 4, 11, 2, 3], \
    [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6], \
    [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10], \
    [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1], \
    [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3], \
    [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1], \
    [3, 11, 6, 3, 6, 0, 0, 6, 4], \
    [6, 4, 8, 11, 6, 8], \
    [7, 10, 6, 7, 8, 10, 8, 9, 10], \
    [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10], \
    [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0], \
    [10, 6, 7, 10, 7, 1, 1, 7, 3], \
    [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7], \
    [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9], \
    [7, 8, 0, 7, 0, 6, 6, 0, 2], \
    [7, 3, 2, 6, 7, 2], \
    [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7], \
    [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7], \
    [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11], \
    [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1], \
    [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6], \
    [0, 9, 1, 11, 6, 7], \
    [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0], \
    [7, 11, 6], \
    [7, 6, 11], \
    [3, 0, 8, 11, 7, 6], \
    [0, 1, 9, 11, 7, 6], \
    [8, 1, 9, 8, 3, 1, 11, 7, 6], \
    [10, 1, 2, 6, 11, 7], \
    [1, 2, 10, 3, 0, 8, 6, 11, 7], \
    [2, 9, 0, 2, 10, 9, 6, 11, 7], \
    [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8], \
    [7, 2, 3, 6, 2, 7], \
    [7, 0, 8, 7, 6, 0, 6, 2, 0], \
    [2, 7, 6, 2, 3, 7, 0, 1, 9], \
    [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6], \
    [10, 7, 6, 10, 1, 7, 1, 3, 7], \
    [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8], \
    [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7], \
    [7, 6, 10, 7, 10, 8, 8, 10, 9], \
    [6, 8, 4, 11, 8, 6], \
    [3, 6, 11, 3, 0, 6, 0, 4, 6], \
    [8, 6, 11, 8, 4, 6, 9, 0, 1], \
    [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6], \
    [6, 8, 4, 6, 11, 8, 2, 10, 1], \
    [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6], \
    [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9], \
    [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3], \
    [8, 2, 3, 8, 4, 2, 4, 6, 2], \
    [0, 4, 2, 4, 6, 2], \
    [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8], \
    [1, 9, 4, 1, 4, 2, 2, 4, 6], \
    [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1], \
    [10, 1, 0, 10, 0, 6, 6, 0, 4], \
    [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3], \
    [10, 9, 4, 6, 10, 4], \
    [4, 9, 5, 7, 6, 11], \
    [0, 8, 3, 4, 9, 5, 11, 7, 6], \
    [5, 0, 1, 5, 4, 0, 7, 6, 11], \
    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5], \
    [9, 5, 4, 10, 1, 2, 7, 6, 11], \
    [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5], \
    [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2], \
    [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6], \
    [7, 2, 3, 7, 6, 2, 5, 4, 9], \
    [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7], \
    [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0], \
    [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8], \
    [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7], \
    [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4], \
    [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10], \
    [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10], \
    [6, 9, 5, 6, 11, 9, 11, 8, 9], \
    [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5], \
    [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11], \
    [6, 11, 3, 6, 3, 5, 5, 3, 1], \
    [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6], \
    [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10], \
    [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5], \
    [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3], \
    [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2], \
    [9, 5, 6, 9, 6, 0, 0, 6, 2], \
    [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8], \
    [1, 5, 6, 2, 1, 6], \
    [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6], \
    [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0], \
    [0, 3, 8, 5, 6, 10], \
    [10, 5, 6], \
    [11, 5, 10, 7, 5, 11], \
    [11, 5, 10, 11, 7, 5, 8, 3, 0], \
    [5, 11, 7, 5, 10, 11, 1, 9, 0], \
    [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1], \
    [11, 1, 2, 11, 7, 1, 7, 5, 1], \
    [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11], \
    [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7], \
    [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2], \
    [2, 5, 10, 2, 3, 5, 3, 7, 5], \
    [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5], \
    [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2], \
    [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2], \
    [1, 3, 5, 3, 7, 5], \
    [0, 8, 7, 0, 7, 1, 1, 7, 5], \
    [9, 0, 3, 9, 3, 5, 5, 3, 7], \
    [9, 8, 7, 5, 9, 7], \
    [5, 8, 4, 5, 10, 8, 10, 11, 8], \
    [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0], \
    [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5], \
    [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4], \
    [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8], \
    [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11], \
    [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5], \
    [9, 4, 5, 2, 11, 3], \
    [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4], \
    [5, 10, 2, 5, 2, 4, 4, 2, 0], \
    [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9], \
    [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2], \
    [8, 4, 5, 8, 5, 3, 3, 5, 1], \
    [0, 4, 5, 1, 0, 5], \
    [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5], \
    [9, 4, 5], \
    [4, 11, 7, 4, 9, 11, 9, 10, 11], \
    [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11], \
    [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11], \
    [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4], \
    [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2], \
    [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3], \
    [11, 7, 4, 11, 4, 2, 2, 4, 0], \
    [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4], \
    [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9], \
    [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7], \
    [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10], \
    [1, 10, 2, 8, 7, 4], \
    [4, 9, 1, 4, 1, 7, 7, 1, 3], \
    [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1], \
    [4, 0, 3, 7, 4, 3], \
    [4, 8, 7], \
    [9, 10, 8, 10, 11, 8], \
    [3, 0, 9, 3, 9, 11, 11, 9, 10], \
    [0, 1, 10, 0, 10, 8, 8, 10, 11], \
    [3, 1, 10, 11, 3, 10], \
    [1, 2, 11, 1, 11, 9, 9, 11, 8], \
    [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9], \
    [0, 2, 11, 8, 0, 11], \
    [3, 2, 11], \
    [2, 3, 8, 2, 8, 10, 10, 8, 9], \
    [9, 10, 2, 0, 9, 2], \
    [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8], \
    [1, 10, 2], \
    [1, 3, 8, 9, 1, 8], \
    [0, 9, 1], \
    [0, 3, 8], \
    []] 


# Process one cube based on the marching cubes algorithm
def process_cube(cube_point, cube_val):

    # Function to linearly interpolate the position where an isosurface
    # cuts an edge between two vertices
    def interp(p1, p2, valp1, valp2):
        return (p1 + (-valp1 / (valp2 - valp1)) * (p2 - p1))
        # Equivalent to:
        # den = abs(valp1) + abs(valp2)
        # return (abs(valp2)/(den))*p1 + (abs(valp1)/den)*p2
   
    # Initialize output triangulation
    vertex = []
    face = []

    # Determine the index into the edge table based on the signs of the
    # implicit function computed at the corners of the cube. The edge
    # table tells us which vertices need to be created. We will create
    # one vertex for each edge crossed by the implicit surface
    cube_index = 0
    if cube_val[0] < 0.0:
        cube_index |= 1
    if cube_val[1] < 0.0:
        cube_index |= 2
    if cube_val[2] < 0.0:
        cube_index |= 4
    if cube_val[3] < 0.0:
        cube_index |= 8
    if cube_val[4] < 0.0:
        cube_index |= 16
    if cube_val[5] < 0.0:
        cube_index |= 32
    if cube_val[6] < 0.0:
        cube_index |= 64
    if cube_val[7] < 0.0:
        cube_index |= 128

    # Cube is entirely in/out of the surface
    # Avoid further processing as no triangulation will be generated
    if edge_table[cube_index] == 0:
        return [vertex, face]

    # Find the edges where the surface intersects the cube and
    # interpolate the corresponding corners to get the vertex at the
    # crossing location
    temp_vertex = [[], [], [], [], [], [], [], [], [], [], [], []]
    if edge_table[cube_index] & 1:
        temp_vertex[0] = \
            interp(cube_point[0], cube_point[1], cube_val[0], cube_val[1])
    if edge_table[cube_index] & 2:
        temp_vertex[1] = \
            interp(cube_point[1], cube_point[2], cube_val[1], cube_val[2])
    if edge_table[cube_index] & 4:
        temp_vertex[2] = \
            interp(cube_point[2], cube_point[3], cube_val[2], cube_val[3])
    if edge_table[cube_index] & 8:
        temp_vertex[3] = \
            interp(cube_point[3], cube_point[0], cube_val[3], cube_val[0])
    if edge_table[cube_index] & 16:
        temp_vertex[4] = \
            interp(cube_point[4], cube_point[5], cube_val[4], cube_val[5])
    if edge_table[cube_index] & 32:
        temp_vertex[5] = \
            interp(cube_point[5], cube_point[6], cube_val[5], cube_val[6])
    if edge_table[cube_index] & 64:
        temp_vertex[6] = \
            interp(cube_point[6], cube_point[7], cube_val[6], cube_val[7])
    if edge_table[cube_index] & 128:
        temp_vertex[7] = \
            interp(cube_point[7], cube_point[4], cube_val[7], cube_val[4])
    if edge_table[cube_index] & 256:
        temp_vertex[8] = \
            interp(cube_point[0], cube_point[4], cube_val[0], cube_val[4])
    if edge_table[cube_index] & 512:
        temp_vertex[9] = \
            interp(cube_point[1], cube_point[5], cube_val[1], cube_val[5])
    if edge_table[cube_index] & 1024:
        temp_vertex[10] = \
            interp(cube_point[2], cube_point[6], cube_val[2], cube_val[6])
    if edge_table[cube_index] & 2048:
        temp_vertex[11] = \
            interp(cube_point[3], cube_point[7], cube_val[3], cube_val[7])

    # Retrieve the triangulation that we should create for this cube
    #
    # Get a list of unique vertex ids that are used in this cube 
    # list(set(...)) eliminates repeated entries in the list of ids
    vertex_index = list(set(tri_table[cube_index]))

    # Append all these unique vertices to the output list of vertices
    for i in vertex_index:
        vertex.append(temp_vertex[i])

    # Create list for remapping vertex ids to this specific case
    remap = [-1 for i in range(12)]
    for i in range(len(vertex_index)):
        remap[vertex_index[i]] = i

    # Append the faces in the cube to the output list of faces
    for i in range(0, len(tri_table[cube_index]), 3):
        face.append([remap[tri_table[cube_index][i+0]],
                     remap[tri_table[cube_index][i+1]],
                     remap[tri_table[cube_index][i+2]]])

    return [vertex, face]


# Run marching cubes algorithm based on an implicit function
#
# Note that this is an entirely new function and not based on the code
# quoted above
def marching_cubes(start, end, num_cubes_per_dim, fun, merge_dup=True):
    """Run the marching cubes algorithm to reconstruct a surface given
    by an implicit function

    Parameters
    ----------
    start : array_like
        3D vector with the starting position along the x, y, and z
        dimensions of the volume to be reconstructed
    end : array_like
        3D vector with the end position along the x, y, and z dimensions
        of the volume to be reconstructed
    num_cubes_per_dim : int
        Number of cubes to create along each dimension. The output mesh
        will be based on a volume subdivided into num_cubes_per_dim^3
        cubes
    fun : function
        Implicit function in the form fun(x) : float, where x is a 3D
        point and the return value represents the approximate signed
        distance of the point to the surface
    merge_dup : boolean
        Perform post-processing to merge duplicated vertices at the
        edges of the cubes and avoid a mesh with boundaries (default
        value is True)

    Returns
    -------
    tm : geomproc.mesh
        A triangle mesh with the reconstructed surface

    See Also
    --------
    geomproc.impsurf

    Examples
    --------
    >>> import geomproc
    >>> import numpy as np
    >>> surf = geomproc.impsurf()
    >>> surf.setup_torus(0.6, 0.3)
    >>> tm = geomproc.marching_cubes(np.array([-1, -1, -1]), np.array([1, 1, 1]), 32, surf.evaluate)
    >>> tm.save('reconstruction.obj')
    """

    # Initialize temporary storage of the mesh
    #
    # We use lists as we need to grow them dynamically, while numpy
    # arrays cannot be grown dynamically as they are stored in
    # contiguous blocks in memory
    vertex = []
    face = []

    # Index of the last vertex added for a cube 
    # We use this index to shift the vertex ids as we add more vertices
    # to the full mesh
    last_index = 0

    # Determine dimensions of each cube based on input parameters
    cube_size = (end - start) / num_cubes_per_dim

    # Process one cube at a time
    # Initialize cube point coordinates and function values
    cube_point = [np.array([0, 0, 0]) for i in range(8)]
    cube_val = [0 for i in range(8)]
    # Go through each cube of the volume
    z = start[2]
    while z < end[2]:
        y = start[1]
        while y < end[1]:
            x = start[0]
            while x < end[0]:
                # Go over the 8 corners of the cube and store the
                # coordinates of the corners and function value at the
                # corners
                point_index = 0
                for k in range(2):
                    # Note that the marching cubes tables assume a
                    # circular ordering of (x, y) positions, so we need
                    # to use this special for loop on a list
                    for shift in [[0, 0], [1, 0], [1, 1], [0, 1]]:
                        # Determine corner coordinates
                        cube_point[point_index] = np.array([\
                            x + shift[0]*cube_size[0], \
                            y + shift[1]*cube_size[1], \
                            z + k*cube_size[2]])
                        # Compute value of implicit function at the
                        # corner
                        cube_val[point_index] = fun(cube_point[point_index])
                        # Increment index for the next corner
                        point_index += 1
                # Now that we have all the function values at the eight
                # corners, process the cube and create a triangulation
                [cube_vertex, cube_face] = process_cube(cube_point, cube_val)
                # Append triangulation to our ongoing mesh
                if len(cube_vertex) > 0:
                    # Append vertices
                    for i in range(len(cube_vertex)):
                        vertex.append(cube_vertex[i])
                    # Append faces 
                    for i in range(len(cube_face)):
                        # Add a shift to the vertex ids, to consider all
                        # the vertices added before
                        for j in range(len(cube_face[i])):
                            cube_face[i][j] += last_index
                        # Fix normals by inverting vertex ordering
                        cube_face[i] = [cube_face[i][0],
                                        cube_face[i][2], cube_face[i][1]]
                        # Append face
                        face.append(cube_face[i])
                    # Increment shift to vertex ids
                    last_index += len(cube_vertex)

                # Increments to continue the while loop
                x += cube_size[0]
            y += cube_size[1]
        z += cube_size[2]

    # Create output mesh
    tm = mesh()
    tm.vertex = np.array(vertex)
    tm.face = np.array(face, dtype=np.int_)

    # Check if we should apply post-processing to merge duplicated
    # vertices
    if merge_dup:
        tm.remove_duplicated_vertices(np.finfo(float).eps*10.0)
    
    # Return mesh
    return tm
