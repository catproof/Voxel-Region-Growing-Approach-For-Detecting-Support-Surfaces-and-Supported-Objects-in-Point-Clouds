#
# GeomProc: geometry processing library in python + numpy
#
# Copyright (c) 2008-2021 Oliver van Kaick <ovankaic@gmail.com>
# under the MIT License.
#
# See file LICENSE.txt for details on copyright licenses.
#
"""This module contains auxiliary functions of the GeomProc geometry
processing library.
"""


import numpy as np
import math
import random


#### Auxiliary functions

# Linearly map values from one range to another
def map_val(val, cmin=0.0, cmax=1.0, mn=1.0, mx=-1.0):
    """Linearly map a set of values to a new range

    Parameters
    ----------
    val : numpy.array_like 
        Array containing the values to be mapped
    cmin : float, optional
        Minimum value for the new range
    cmax : float, optional
        Maximum value for the new range
    mn : float, optional
        The value to be considered as the minimum in the range of 'val'.
        If this parameter is not provided, it is computed as min(val).
        Setting a minimum that is different from the actual data minimum
        allows to clamp values that are too small. Note that, if 'mn' is
        provided, then both 'mn' and 'mx' need to be provided
    mx : float, optional
        The value to be considered the maximum in the range of 'val'. If
        this parameter is not provided, it is computed as max(val).
        Setting a maximum that is different from the actual data maximum
        allows to clamp values that are too big. Note that, if 'mn' is
        provided, then both 'mn' and 'mx' need to be provided

    Returns
    -------
    new_val : numpy.array_like 
        Array containing 'val' mapped to the new range [cmin, cmax]

    Notes
    -----
    The purpose of this function is to linearly map a set of values
    'val' in the range [mn, mx] to a new set of values 'new_val' in the
    range [cmin, cmax]. This can be utilized, for example, to map data
    values to color values.
    """
 
    # Check if minimum and maximum were provided:
    if mx < mn:
        mx = val.max()
        mn = val.min()

    # Clamp values
    temp = val.copy()
    temp[temp > mx] = mx
    temp[temp < mn] = mn

    # Map values
    return cmin + ((temp - mn)/(mx - mn))*(cmax - cmin)


# Map HSV color values to RGB
def hsv2rgb(hsv):
    """Map an HSV color into an RGB color

    Parameters
    ----------
    hsv : numpy.array_like
        Array with three float values representing the Hue, Saturation,
        and Value components of the color. Each component should be a
        value between 0 and 1

    Returns
    -------
    rgb: numpy.array_like
        Array with three float values representing the corresponding
        Red, Green, and Blue components of the color. Each component is
        a value between 0 and 1

    Examples
    --------
    >>> import geomproc
    >>> rgb = geomproc.hsv2rgb([0.5, 0.8, 0.8])
    """
 
    # Output RGB color
    out = [0, 0, 0]

    # Perform mapping with this rather obscure algorithm
    h = hsv[0]*360.0
    if h >= 360.0:
        h = 0.0
    h /= 60.0
    i = math.floor(h)
    f = h - i
    p = hsv[2] * (1.0 - hsv[1])
    q = hsv[2] * (1.0 - (hsv[1] * f))
    t = hsv[2] * (1.0 - (hsv[1] * (1.0 - f)))

    if i == 0:
        out[0] = hsv[2]
        out[1] = t
        out[2] = p
    elif i == 1:
        out[0] = q
        out[1] = hsv[2]
        out[2] = p
    elif i == 2:
        out[0] = p
        out[1] = hsv[2]
        out[2] = t
    elif i == 3:
        out[0] = p
        out[1] = q
        out[2] = hsv[2]
    elif i == 4:
        out[0] = t
        out[1] = p
        out[2] = hsv[2]
    else:
        out[0] = hsv[2]
        out[1] = p
        out[2] = q

    return out


def rotation_matrix(ang, axs):
    """Create a rotation matrix for a rotation of an angle around an axis
    
    Parameters
    ----------
    axis : array_like
        Rotation axis: a 3D vector
    angle : float
        Rotation angle in radians

    Returns
    -------
    matrix : array_like
        4x4 rotation matrix in column-major form
    """

    # Init 3x3 rotation matrix
    R = np.zeros((3, 3))

    # Normalize axis vector and get its components
    axs /= np.linalg.norm(axs)
    x = axs[0]
    y = axs[1]
    z = axs[2]

    # Compute sin and cos of rotation angle
    cosa = math.cos(ang)
    sina = math.sin(ang)

    # Create matrix
    # For an explanation, see
    # http://en.wikipedia.org/wiki/Rotation_matrix
    R[0,0] = cosa + (1 - cosa)*x*x
    R[0,1] = (1 - cosa)*x*y - sina*z
    R[0,2] = (1 - cosa)*x*z + sina*y

    R[1,0] = (1 - cosa)*y*x + sina*z
    R[1,1] = cosa + (1 - cosa)*y*y
    R[1,2] = (1 - cosa)*y*z - sina*x

    R[2,0] = (1 - cosa)*z*x - sina*y
    R[2,1] = (1 - cosa)*z*y + sina*x
    R[2,2] = cosa + (1 - cosa)*z*z

    return R


# Randomly sample a point inside of a triangle
def random_triangle_sample(tm, index):
    """Randomly sample a point inside of a triangle without geometric bias
    
    Parameters
    ----------
    tm : geomproc.mesh
        Triangle mesh
    index : float
        Index of the triangle in the mesh to be sampled

    Returns
    -------
    point : array_like
        A random 3D point on the surface of the triangle and inside its
        boundaries
    """

    # Reference: Osada et al., "Shape distributions", TOG 2002, page 814.

    # Get two random numbers
    r1 = random.random()
    r2 = random.random()

    # Get three vertices of the selected face
    v0 = tm.vertex[tm.face[index, 0], :]
    v1 = tm.vertex[tm.face[index, 1], :]
    v2 = tm.vertex[tm.face[index, 2], :]

    # Get random point without bias
    point = (1 - math.sqrt(r1))*v0 + math.sqrt(r1)*(1 - r2)*v1 + math.sqrt(r1)*r2*v2
    return point


# Euclidean distance between two 3D points stored as lists
def distance(a, b):
    """Euclidean distance between two points
    
    Parameters
    ----------
    a : list of array_like
        First 3D point
    b : list or array_like
        Second 3D point

    Returns
    -------
    d : float
        Euclidean distance between a and b
    """

    return math.sqrt((a[0] - b[0])*(a[0] - b[0]) +
                     (a[1] - b[1])*(a[1] - b[1]) +
                     (a[2] - b[2])*(a[2] - b[2]))
