#
# GeomProc: geometry processing library in python + numpy
#
# Copyright (c) 2008-2021 Oliver van Kaick <ovankaic@gmail.com>
# under the MIT License.
#
# See file LICENSE.txt for details on copyright licenses.
#
"""This module contains functions for creating geometric datasets of the
GeomProc geometry processing library.
"""


import numpy as np
import math
import random

from .mesh import *
from .pcloud import *
from .misc import *


#### Functions for creating geometric datasets

# Create a sphere
def create_sphere(radius, num_theta_samples, num_phi_samples):
    """Create a 3D model of a sphere

    Parameters
    ----------
    radius : float
        Radius of the sphere
    num_theta_samples : int
        Number of uniform angular samples along the equator of the sphere
    num_phi_samples : int
        Number of uniform angular samples along the meridians of the sphere

    Returns
    -------
    tm : mesh
        Mesh object representing the sphere

    Notes
    -----
    The function creates a triangle mesh representing a sphere with
    given radius and centered at the origin. The sphere is built from a
    standard parameterization based on two angles theta and phi. The
    function also computes the analytical vertex normal vectors and
    curvature values of the surface, stored in the 'vnormal' and 'curv'
    attributes of the object. See the help of 'mesh.compute_curvature'
    for information on the 'curv' attribute.

    See Also
    --------
    geomproc.mesh.mesh.compute_curvature

    Examples
    --------
    >>> import geomproc
    >>> tm = geomproc.create_sphere(1.0, 30, 30)
    """
    
    if radius == 0:
        tm = mesh()
        tm.vertex = np.zeros((1, 3))
        return tm

    # Calculate size of mesh
    num_verts = num_theta_samples*(num_phi_samples-2) + 2
    num_faces = num_theta_samples*(num_phi_samples-3)*2 + num_theta_samples*2

    # Initialize output mesh
    tm = mesh()
    tm.vertex = np.zeros((num_verts, 3))
    tm.vnormal = np.zeros((num_verts, 3))
    tm.curv = np.zeros((num_verts, 5))
    tm.face = np.zeros((num_faces, 3), dtype=np.int_)

    # Create vertices
    vindex = 0

    # Create central vertices
    for j in range(1, num_phi_samples-1):
        phi = -(math.pi/2) + math.pi*j/(num_phi_samples-1)
        for i in range(num_theta_samples):
            theta = 2*math.pi*i/num_theta_samples
            # Define vertex normal
            nrm = np.array([math.cos(theta)*math.cos(phi), \
                            math.sin(theta)*math.cos(phi), \
                            math.sin(phi)])
            nrm /= np.linalg.norm(nrm)
            tm.vnormal[vindex, :] = nrm
            # Define vertex position
            tm.vertex[vindex, :] = radius*nrm
            # Define curvatures
            tm.curv[vindex, 0] = 1/radius
            tm.curv[vindex, 1] = 1/radius
            tm.curv[vindex, 2] = 1/radius
            tm.curv[vindex, 3] = 1/(radius*radius)
            # Next vertex
            vindex = vindex + 1

    # Create poles
    for i in [-1, 1]:
        # Define vertex normal
        nrm = np.array([0, 0, i])
        tm.vnormal[vindex, :] = nrm
        # Define vertex position
        tm.vertex[vindex, :] = radius*nrm
        # Define curvatures
        tm.curv[vindex, 0] = 1/radius
        tm.curv[vindex, 1] = 1/radius
        tm.curv[vindex, 2] = 1/radius
        tm.curv[vindex, 3] = 1/(radius*radius)
        # Next vertex
        vindex = vindex + 1

    # Create faces
    findex = 0

    # Central part of the mesh
    for j in range(num_phi_samples-3):
        for i in range(num_theta_samples):
            # (i, j), (i+1, j), (i, j+1)
            tm.face[findex, :] = [j*num_theta_samples + \
                                  (i % num_theta_samples), \
                                  j*num_theta_samples + \
                                  ((i+1) % num_theta_samples), \
                                  (j+1)*num_theta_samples + \
                                  (i % num_theta_samples)]
            findex = findex + 1
            # (i+1, j), (i+1, j+1), (i, j+1)
            tm.face[findex, :] = [j*num_theta_samples + \
                                  ((i+1) % num_theta_samples), \
                                  (j+1)*num_theta_samples + \
                                  ((i+1) % num_theta_samples), \
                                  (j+1)*num_theta_samples + \
                                  (i % num_theta_samples)]
            findex = findex + 1

    # Attach poles
    for i in range(num_theta_samples):
        # (i, j), (i+1, j), (pole) with j = 0
        pole = vindex-2
        tm.face[findex, :] = np.array([(i % num_theta_samples), \
                                      pole, \
                                      ((i+1) % num_theta_samples)])
        findex = findex + 1
        # (i, j), (i+1, j), (pole)
        pole = vindex-1
        j = num_phi_samples-3
        tm.face[findex, :] = np.array([j*num_theta_samples + \
                                       (i % num_theta_samples), \
                                      j*num_theta_samples + \
                                      ((i+1) % num_theta_samples), \
                                      pole])
        findex = findex + 1

    return tm


# Create a torus
def create_torus(loop_radius, tube_radius, num_loop_samples, num_tube_samples):
    """Create a 3D model of a torus

    Parameters
    ----------
    loop_radius : float
        Radius of the loop (larger circle) of the torus
    tube_radius : float
        Radius of the tube (smaller circle) of the torus
    num_loop_samples : int
        Number of uniform angular samples along the loop of the torus
    num_tube_samples : int
        Number of uniform angular samples along the tube of the torus

    Returns
    -------
    tm : mesh
        Mesh object representing the torus

    Notes
    -----
    The function creates a triangle mesh representing a torus with given
    radii and centered at the origin. The torus is built from a large
    loop with small circles around the loop to form the tube of the
    torus. The function also computes the analytical vertex normal
    vectors and curvature values of the surface, stored in the 'vnormal'
    and 'curv' attributes of the object.  See the help of
    'mesh.compute_curvature' for information on the 'curv' attribute.

    See Also
    --------
    geomproc.mesh.mesh.compute_curvature

    Examples
    --------
    >>> import geomproc
    >>> tm = geomproc.create_torus(1.0, 0.33, 90, 30)
    """

    # Reference on torus equation:
    # https://www.math.hmc.edu/~gu/curves_and_surfaces/surfaces/torus.html

    # Initialize output
    tm = mesh()
    tm.vertex = np.zeros((num_loop_samples*num_tube_samples, 3))
    tm.vnormal = np.zeros((num_loop_samples*num_tube_samples, 3))
    tm.curv = np.zeros((num_loop_samples*num_tube_samples, 5))
    tm.face = np.zeros((num_loop_samples*num_tube_samples*2, 3), dtype=np.int_)

    # Create vertices
    vindex = 0
    for i in range (num_loop_samples):
        theta = 2*math.pi*i/num_loop_samples
        loop_center = [loop_radius*math.cos(theta), loop_radius*math.sin(theta), 0]
        for j in range(num_tube_samples):
            # Define vertex normal
            phi = 2*math.pi*j/num_tube_samples
            nrm = [math.cos(theta)*math.cos(phi), \
                   math.sin(theta)*math.cos(phi), \
                   math.sin(phi)]
            nrm /= np.linalg.norm(nrm)
            tm.vnormal[vindex, :] = nrm
            # Define vertex position
            tm.vertex[vindex, :] = loop_center + nrm*tube_radius
            # Define curvatures
            # Mean curvature
            H = (loop_radius + 2*tube_radius*math.cos(phi))/ \
                (2*tube_radius*(loop_radius + tube_radius*math.cos(phi)))
            # Gaussian curvature
            K = math.cos(phi)/ \
                (tube_radius*(loop_radius + tube_radius*math.cos(phi)))
            # Derive minimum and maximum curvature from mean and Gaussian
            tm.curv[vindex, 2] = H
            tm.curv[vindex, 3] = K
            tm.curv[vindex, 0] = H - math.sqrt(H*H - K)
            tm.curv[vindex, 1] = H + math.sqrt(H*H - K)
            if tm.curv[vindex, 1] < tm.curv[vindex, 0]:
                temp = tm.curv[vindex, 0]
                tm.curv[vindex, 0] = model.curv[vindex, 1]
                tm.curv[vindex, 1] = temp
            vindex = vindex + 1

    # Create faces
    findex = 0
    for i in range (num_loop_samples):
        for j in range(num_tube_samples):
            # (i+1, j), (i, j+1), (i, j)
            tm.face[findex, :] = \
                [((i+1) % num_loop_samples)*num_tube_samples + j, \
                 i*num_tube_samples + ((j+1) % num_tube_samples), \
                 i*num_tube_samples + j]
            findex = findex + 1
            # (i+1, j), (i+1, j+1), (i, j+1)
            tm.face[findex, :] = \
                [((i+1) % num_loop_samples)*num_tube_samples + j, \
                 ((i+1) % num_loop_samples)*num_tube_samples + ((j+1) % num_tube_samples), \
                i*num_tube_samples + ((j+1) % num_tube_samples)]
            findex = findex + 1
    
    return tm


# Create a cylinder
def create_cylinder(radius, height, num_circle_samples, num_height_samples, closed=False):
    """Create a 3D model of a cylinder

    Parameters
    ----------
    radius : float
        Radius of the cylinder
    height : float
        Height of the cylinder
    num_circle_samples: int
        Number of uniform angular samples along each circle of the cylinder
    num_height_samples : int
        Number of samples along the height of the cylinder
    closed : boolean, optional (default False)
        Whether the cylinder should be closed at the ends or not

    Returns
    -------
    tm : mesh
        Mesh object representing the cylinder

    Notes
    -----
    The cylinder is built so that the circles are parallel to the XY
    plane and the cylinder extends from -height/2 to height/2. The
    function also computes the analytical vertex normal vectors and
    curvature values of the surface, stored in the 'vnormal' and 'curv'
    fields of the model. See the help of 'mesh.compute_curvature' for
    information on the 'curv' attribute.

    See Also
    --------
    geomproc.mesh.mesh.compute_curvature

    Examples
    --------
    >>> import geomproc
    >>> tm = geomproc.create_cylinder(0.5, 1, 60, 10)
    """

    # Initialize output
    tm = mesh()
    if closed:
        tm.vertex = np.zeros((num_circle_samples*num_height_samples + 2, 3))
        tm.vnormal = np.zeros((num_circle_samples*num_height_samples + 2, 3))
        tm.curv = np.zeros((num_circle_samples*num_height_samples + 2, 5))
        tm.face = np.zeros((num_circle_samples*(num_height_samples-1)*2 + \
                            num_circle_samples*2, 3), dtype=np.int_)
    else:
        tm.vertex = np.zeros((num_circle_samples*num_height_samples, 3))
        tm.vnormal = np.zeros((num_circle_samples*num_height_samples, 3))
        tm.curv = np.zeros((num_circle_samples*num_height_samples, 5))
        tm.face = np.zeros((num_circle_samples*(num_height_samples-1)*2, 3), dtype=np.int_)

    # Create vertices
    vindex = 0
    for i in range(num_height_samples):
        h = (-height/2) + height*i/(num_height_samples-1)
        for j in range(num_circle_samples):
            # Define vertex position
            theta = 2*math.pi*j/num_circle_samples
            tm.vertex[vindex, :] = [radius*math.cos(theta), \
                                    radius*math.sin(theta), h]
            # Define normal
            nrm = [math.cos(theta), math.sin(theta), 0]
            nrm /= np.linalg.norm(nrm)
            tm.vnormal[vindex, :] = nrm
            # Define curvatures
            tm.curv[vindex, 0] = 0
            tm.curv[vindex, 1] = 1/radius
            tm.curv[vindex, 2] = 1/(2*radius)
            tm.curv[vindex, 3] = 0
            vindex = vindex + 1

    # Create end poles, if needed
    if closed:
        # Add a pole
        tm.vertex[vindex, :] = [0, 0, -height/2.0]
        # Define normal
        nrm = [0, 0, -1]
        nrm /= np.linalg.norm(nrm)
        tm.vnormal[vindex, :] = nrm
        # Define curvatures
        tm.curv[vindex, 0] = 0
        tm.curv[vindex, 1] = 0
        tm.curv[vindex, 2] = 0
        tm.curv[vindex, 3] = 0
        vindex = vindex + 1

        # Add another pole
        tm.vertex[vindex, :] = [0, 0, height/2.0]
        # Define normal
        nrm = [0, 0, 1]
        nrm /= np.linalg.norm(nrm)
        tm.vnormal[vindex, :] = nrm
        # Define curvatures
        tm.curv[vindex, 0] = 0
        tm.curv[vindex, 1] = 0
        tm.curv[vindex, 2] = 0
        tm.curv[vindex, 3] = 0
        vindex = vindex + 1

    # Create faces
    findex = 0
    for i in range(num_height_samples-1):
        for j in range(num_circle_samples):
            # (i+1, j), (i, j), (i, j+1)
            tm.face[findex, :] = \
                [(i+1)*num_circle_samples + j, \
                 i*num_circle_samples + j, \
                 i*num_circle_samples + ((j+1) % num_circle_samples)]
            findex = findex + 1
            # (i+1, j), (i, j+1), (i+1, j+1)
            tm.face[findex, :] = \
                [(i+1)*num_circle_samples + j, \
                 i*num_circle_samples + ((j+1) % num_circle_samples), \
                 (i+1)*num_circle_samples + ((j+1) % num_circle_samples)]
            findex = findex + 1

    # Close ends of cylinder, if needed:
    if closed:
        for j in range(num_circle_samples):
            tm.face[findex, :] = \
                [j, \
                 vindex-2, \
                 ((j+1) % num_circle_samples)]
            findex = findex + 1
        for j in range(num_circle_samples):
            tm.face[findex, :] = \
                [vindex-1, \
                 (num_height_samples-1)*num_circle_samples + j, \
                 (num_height_samples-1)*num_circle_samples +\
                        ((j+1) % num_circle_samples)]
            findex = findex + 1

    return tm


# Create a cone
def create_cone(radius, height, num_circle_samples, num_height_samples):
    """Create a 3D model of a cone

    Parameters
    ----------
    radius : float
        Radius of the cone at the base
    height : float
        Height of the cone
    num_circle_samples: int
        Number of uniform angular samples along each circle of the cone
    num_height_samples : int
        Number of samples along the height of the cone

    Returns
    -------
    tm : mesh
        Mesh object representing the cone

    Notes
    -----
    The cone is built so that the circles are parallel to the XY plane
    and the cone extends from 0 to height. The function also computes
    the analytical vertex normal vectors and curvature values of the
    surface, stored in the 'vnormal' and 'curv' fields of the model. See
    the help of 'mesh.compute_curvature' for information on the 'curv'
    attribute.

    See Also
    --------
    geomproc.mesh.mesh.compute_curvature

    Examples
    --------
    >>> import geomproc
    >>> tm = geomproc.create_cone(1, 1, 60, 10)
    """

    # Initialize output
    tm = mesh()
    tm.vertex = np.zeros((num_circle_samples*(num_height_samples-1) + 1, 3))
    tm.vnormal = np.zeros((num_circle_samples*(num_height_samples-1) + 1, 3))
    tm.curv = np.zeros((num_circle_samples*(num_height_samples-1) + 1, 5))
    tm.face = np.zeros((num_circle_samples*((num_height_samples-2)*2 + 1), 3), dtype=np.int_)

    # Create vertices
    vindex = 0
    for i in range(num_height_samples-1):
        h = height*i/(num_height_samples-1)
        for j in range(num_circle_samples):
            # Define vertex position
            theta = 2*math.pi*j/num_circle_samples
            scaling = (height - h)/height
            tm.vertex[vindex, :] = [scaling*radius*math.cos(theta), \
                                    scaling*radius*math.sin(theta), h]
            # Define normal
            nrm = [math.cos(theta), math.sin(theta), 0]
            nrm /= np.linalg.norm(nrm)
            tm.vnormal[vindex, :] = nrm
            # Define curvatures
            tm.curv[vindex, 0] = 0
            tm.curv[vindex, 1] = 1/(scaling*radius)
            tm.curv[vindex, 2] = 1/(2*scaling*radius)
            tm.curv[vindex, 3] = 0
            vindex = vindex + 1
    # Add apex
    tm.vertex[vindex, :] = [0, 0, height]
    # Define normal
    nrm = [0, 0, 1]
    nrm /= np.linalg.norm(nrm)
    tm.vnormal[vindex, :] = nrm
    # Define curvatures
    tm.curv[vindex, 0] = 0
    tm.curv[vindex, 1] = float('inf') # math.inf
    tm.curv[vindex, 2] = float('inf')
    tm.curv[vindex, 3] = 0
    vindex = vindex + 1

    # Create faces
    findex = 0
    for i in range(num_height_samples-2):
        for j in range(num_circle_samples):
            # (i+1, j), (i, j), (i, j+1)
            tm.face[findex, :] = \
                [(i+1)*num_circle_samples + j, \
                 i*num_circle_samples + j, \
                 i*num_circle_samples + ((j+1) % num_circle_samples)]
            findex = findex + 1
            # (i+1, j), (i, j+1), (i+1, j+1)
            tm.face[findex, :] = \
                [(i+1)*num_circle_samples + j, \
                 i*num_circle_samples + ((j+1) % num_circle_samples), \
                 (i+1)*num_circle_samples + ((j+1) % num_circle_samples)]
            findex = findex + 1
    # Connect cone to apex
    apex = tm.vertex.shape[0]-1
    for j in range (num_circle_samples):
        # (i, j), (i, j+1), apex
        tm.face[findex, :] = \
            [i*num_circle_samples + j, \
             i*num_circle_samples + ((j+1) % num_circle_samples), \
             apex]
        findex = findex + 1

    return tm


# Create an open surface
def create_open_surface(num_x_samples, num_y_samples, surf_type):
    """Create a 3D open surface based on a mathematical function of two variables

    Parameters
    ----------
    num_x_samples : int
        Number of samples to add along the x axis
    num_y_samples : int 
        Number of samples to add along the y axis
    surf_type : int
        Surface type according to the following codes:
            * 0: flat surface f(x, y) = 0
            * 1: hyperbolic paraboloid (simple saddle) f(x, y) = x*y
            * 2: monkey saddle f(x, y) = x^3-3*x*y^2
            * 3: hemisphere f(x, y) = sqrt(1 - x^2 - y^2)

    Returns
    -------
    tm : mesh
        Mesh object representing the surface

    Notes
    -----
    The function creates an open surface by defining a 2D grid extending
    from (-1, -1) to (1, 1) and computing a function of two variables at
    the grid vertices. The function also computes the analytical vertex
    normal vectors and curvature values of the surfaces, stored in the
    'vnormal' and 'curv' fields of the model. See the help of
    'mesh.compute_curvature' for information on the 'curv' attribute.

    See Also
    --------
    geomproc.mesh.mesh.compute_curvature

    Examples
    --------
    >>> import geomproc
    >>> tm = geomproc.create_open_surface(30, 30, 1)
    """

    # References for these surfaces:
    # Note that the mean curvature in these definitions is often
    # provided with inverted sign
    # http://cgtools.net/data_generation_meshes.php
    # http://mathworld.wolfram.com/HyperbolicParaboloid.html
    # http://mathworld.wolfram.com/MonkeySaddle.html

    # Initialize output
    tm = mesh()
    tm.vertex = np.zeros((num_x_samples*num_y_samples, 3))
    tm.vnormal = np.zeros((num_x_samples*num_y_samples, 3))
    tm.curv = np.zeros((num_x_samples*num_y_samples, 5))
    tm.face = np.zeros(((num_x_samples-1)*(num_y_samples-1)*2, 3), dtype=np.int_)

    # Create vertices
    vindex = 0
    for i in range(num_y_samples):
        y = -1 + 2*i/(num_y_samples-1)
        for j in range (num_x_samples):
            x = -1 + 2*j/(num_x_samples-1)
            if surf_type == 0:
                # Flat surface
                tm.vertex[vindex, :] = [x, y, 0]
                tm.vnormal[vindex, :] = [0, 0, 1]
                H = 0
                K = 0
            elif surf_type == 1:
                # Hyperbolic paraboloid (simple saddle)
                tm.vertex[vindex, :] = [x, y, x*y]
                tm.vnormal[vindex, :] = [-y, -x, 1]
                H = (x*y)/math.sqrt((1 + x*x + y*y)**3)
                K = -1/((1 + x*x + y*y)**2)
            elif surf_type == 2:
                # Monkey saddle
                tm.vertex[vindex, :] = [x, y, x*x*x -3*x*y*y]
                tm.vnormal[vindex, :] = [3*(y*y - x*x), 6*x*y, 1]
                H = -(27*x*(-x**4 + 2*x*x*y*y+3*y**4))/math.sqrt((1 + 9*(x*x + y*y)**2)**3)
                K = -(36*(x*x + y*y))/((1 + 9*(x*x + y*y)**2)**2)
            elif surf_type == 3:
                # Hemisphere
                radius = 1
                val = radius*radius -x*x -y*y
                z = 0
                if val > 0:
                    z = math.sqrt(val)
                tm.vertex[vindex, :] = [x, y, z]
                tm.vnormal[vindex, :] = tm.vertex[vindex, :]
                H = 1/radius
                K = 1/(radius*radius)
            # Normalize normal vector
            tm.vnormal[vindex, :] /= np.linalg.norm(tm.vnormal[vindex, :])
            # Assign curvatures
            tm.curv[vindex, 2] = H
            tm.curv[vindex, 3] = K
            tm.curv[vindex, 0] = H - math.sqrt(H*H - K)
            tm.curv[vindex, 1] = H + math.sqrt(H*H - K)
            if tm.curv[vindex, 1] < tm.curv[vindex, 0]:
                temp = tm.curv[vindex, 0]
                tm.curv[vindex, 0] = tm.curv[vindex, 1]
                tm.curv[vindex, 1] = temp
            vindex = vindex + 1

    # Create faces
    findex = 0
    for i in range(num_y_samples-1):
        for j in range (num_x_samples-1):
            # (i+1, j), (i, j), (i, j+1), 
            tm.face[findex, :] = \
                [(i+1)*num_x_samples + j, \
                 i*num_x_samples + j, \
                 i*num_x_samples + j + 1]
            findex = findex + 1
            # (i+1, j), (i, j+1), (i+1, j+1), 
            tm.face[findex, :] = \
                [(i+1)*num_x_samples + j, \
                 i*num_x_samples + j + 1, \
                 (i+1)*num_x_samples + j + 1]
            findex = findex + 1

    return tm


def create_subdivided_cube(length, num_samples):
    """Create a 3D model of a subdivided cube

    Parameters
    ----------
    len : float
        Length of the cube along each dimension
    num_samples : int
        Number of samples along each dimension

    Returns
    -------
    tm : mesh
        Mesh object representing the cube

    Notes
    -----
    The function creates a cube in the range [0, 0, 0] to [length,
    length, length], and subdivides it according to the given number of
    samples. The function also computes the normals of the vertices. To
    ensure that normals are meaningful, the cube is created without
    sharing vertices at the edges.

    Examples
    --------
    >>> import geomproc
    >>> tm = geomproc.create_subdivided_cube(1, 10)
    """
 
    # Initialize output
    tm = mesh()
    tm.vertex = np.zeros((num_samples*num_samples*6, 3))
    tm.vnormal = np.zeros((num_samples*num_samples*6, 3))
    tm.face = np.zeros(((num_samples-1)*(num_samples-1)*6*2, 3), dtype=np.int_)

    # Rotation matrices we will use
    angle = math.pi/2
    rot1 = np.array([[ math.cos(angle), 0, math.sin(angle)], \
                     [               0, 1,               0], \
                     [-math.sin(angle), 0, math.cos(angle)]])
    angle = math.pi
    rot2 = np.array([[ math.cos(angle), 0, math.sin(angle)],
                     [               0, 1,               0],
                     [-math.sin(angle), 0, math.cos(angle)]])
    angle = 3*math.pi/2
    rot3 = np.array([[ math.cos(angle), 0, math.sin(angle)],
                     [               0, 1,               0], 
                     [-math.sin(angle), 0, math.cos(angle)]])
    angle = -math.pi/2
    rot4 = np.array([[1,                0,                0],
                      [0, math.cos(angle), -math.sin(angle)],
                      [0, math.sin(angle),  math.cos(angle)]])
    angle = math.pi/2
    rot5 = np.array([[1,                0,                0],
                      [0, math.cos(angle), -math.sin(angle)],
                      [0, math.sin(angle),  math.cos(angle)]])

    # Create vertices
    vindex = 0
    # Offset of each vertex
    length = float(length)
    factor = length/(num_samples-1)
    # For each face of the cube
    for f in range (6):
        # Create the current face
        for i in range(num_samples):
            for j in range(num_samples):
                # Define initial vertex position
                pos = np.array([factor*j, factor*i, 0])
                pos = pos.T
                # Rotate vertex and define normal
                if f == 0:
                    # Don't modify position
                    # Define normal
                    tm.vnormal[vindex, :] = [0, 0, 1]
                elif f == 1:
                    pos = np.dot(rot1, pos) + np.array([length, 0, 0]).T
                    tm.vnormal[vindex, :] = [1, 0, 0]
                elif f == 2:
                    pos = np.dot(rot2, pos) + np.array([length, 0, -length]).T
                    tm.vnormal[vindex, :] = [1, 0, 0]
                elif f == 3:
                    pos = np.dot(rot3, pos) + np.array([0, 0, -length]).T
                    tm.vnormal[vindex, :] = [1, 0, 0]
                elif f == 4:
                    pos = np.dot(rot4, pos) + np.array([0, length, 0]).T
                    tm.vnormal[vindex, :] = [1, 0, 0]
                elif f == 5:
                    pos = np.dot(rot5, pos) + np.array([0, 0, -length]).T
                    tm.vnormal[vindex, :] = [1, 0, 0]
                # Assign final vertex position
                tm.vertex[vindex, :] = pos.T
                # Increment vertex index
                vindex = vindex + 1

    # Create triangles
    findex = 0
    # For each face of the cube
    for f in range(6):
        # Offset for starting vertex index
        offset = f*num_samples*num_samples
        # Create two triangles
        for i in range(num_samples-1):
            for j in range(num_samples-1):
                # (i+1, j), (i, j), (i, j+1)
                tm.face[findex, :] = \
                    [offset + (i+1)*num_samples + j, \
                     offset + i*num_samples + j, \
                     offset + i*num_samples + j + 1]
                findex = findex + 1
                # (i+1, j), (i, j+1), (i+1, j+1)
                tm.face[findex, :] = \
                    [offset + (i+1)*num_samples + j, \
                     offset + i*num_samples + j + 1, \
                     offset + (i+1)*num_samples + j + 1]
                findex = findex + 1

    return tm


def create_simple_cube():
    """Create a 3D model of a simple cube

    Parameters
    ----------
    None

    Returns
    -------
    tm : mesh
        Mesh object representing the cube

    Notes
    -----
    The function creates the simplest possible model of a cube: a mesh
    with 8 (shared) vertices and 12 triangles (6 faces). The coordinates
    on each side go from 0 to 1.

    Examples
    --------
    >>> import geomproc
    >>> tm = geomproc.create_simple_cube()
    """
 
    # Initialize output
    tm = mesh()
    tm.vertex = np.array([ \
        [0.0, 0.0, 1.0], \
        [1.0, 0.0, 1.0], \
        [0.0, 1.0, 1.0], \
        [1.0, 1.0, 1.0], \
        [1.0, 0.0, 0.0], \
        [1.0, 1.0, 0.0], \
        [0.0, 0.0, 0.0], \
        [0.0, 1.0, 0.0]])
    tm.face = np.array([
        [2, 0, 1], \
        [2, 1, 3], \
        [3, 1, 4], \
        [3, 4, 5], \
        [5, 4, 6], \
        [5, 6, 7], \
        [7, 6, 0], \
        [7, 0, 2], \
        [7, 2, 3], \
        [7, 3, 5], \
        [0, 6, 4], \
        [0, 4, 1]])
    return tm


def create_points(pos, radius=0.03, color=[1, 0, 0]):
    """Create geometry to represent a set of points

    Parameters
    ----------
    pos : array_like
        An array of shape (n, 3), representing n 3D points
    radius : float
        Radius of the sphere representing each point (the default value
        is 0.03)
    color : array_like
        Color of each point (the default value is red)

    Returns
    -------
    tm : mesh
        Mesh object representing the points

    Notes
    -----
    The function creates a mesh composed of multiple small spheres that
    represent a set of points, where each sphere is centered at one of
    the given points and has the specified radius and color.

    Examples
    --------
    >>> import geomproc
    >>> tm = geomproc.create_sphere(1.0, 30, 30)
    >>> pc = tm.sample(1000)
    >>> pnts = geomproc.create_points(pc.point)
    >>> pnts.save('samples.obj')
    """
 
    # Create a canonical sphere with origin at [0, 0, 0]
    sphere = create_sphere(radius, 8, 8)
    temp_sphere = sphere.copy()

    # Create mesh to store the geometry of the sphere, with one
    # entire sphere per entry of 'pos'
    tm = mesh()
    # Create data arrays
    num_vertex = pos.shape[0]*sphere.vertex.shape[0]
    num_face = pos.shape[0]*sphere.face.shape[0]
    tm.vertex = np.zeros((num_vertex, 3))
    tm.vcolor = np.zeros((num_vertex, 3))
    tm.face = np.zeros((num_face, 3), dtype=np.int_)

    # Add a sphere for each point
    vertex_index = 0
    face_index = 0
    for i in range(pos.shape[0]):
        # Retrieve saved geometry of sphere
        temp_sphere.vertex[:] = sphere.vertex[:]
        # Apply translation
        temp_sphere.vertex += pos[i, :]
        # Add geometry of cylinder to mesh
        tm.vertex[vertex_index:(vertex_index + temp_sphere.vertex.shape[0]), :] =\
            temp_sphere.vertex
        tm.vcolor[vertex_index:(vertex_index + temp_sphere.vertex.shape[0]), :] =\
            color
        tm.face[face_index:(face_index + temp_sphere.face.shape[0]), :] = \
            temp_sphere.face + vertex_index
        vertex_index += temp_sphere.vertex.shape[0]
        face_index += temp_sphere.face.shape[0]

    return tm


def create_vectors(pos, vect, radius=0.03, length=0.05, color=[1, 0, 0]):
    """Create geometry to represent a set of vectors

    Parameters
    ----------
    pos : array_like
        An array of shape (n, 3), representing the origin of each vector
    vect: array_like
        An array of shape (n, 3), representing n 3D vectors
    radius : float
        Radius of the cylinder representing each vector (the default
        value is 0.03)
    length : float
        Length of the cylinder representing each vector (the default
        value is 0.05)
    color : array_like
        Color of each vector (the default value is red)

    Returns
    -------
    tm : mesh
        Mesh object representing the vectors

    Notes
    -----
    The function creates a mesh composed of multiple small cylinders
    that represent a set of vectors. Each cylinder starts at the
    provided origin for the vector and is oriented according to the
    provided vector.

    Examples
    --------
    >>> import geomproc
    >>> tm = geomproc.create_sphere(1.0, 30, 30)
    >>> tm.compute_vertex_and_face_normals()
    >>> vn = geomproc.create_vectors(tm.vertex, tm.vnormal, color=[0, 0, 1])
    >>> vn.save('vertex_normals.obj')
    """
 
    # Create a canonical cylinder with origin at [0, 0, 0] and
    # pointing to [0, 0, length]
    cyl = create_cylinder(radius, 1, 8, 2, True)
    cyl.vertex += np.array([0.0, 0.0, 0.5])
    cyl.vertex *= length
    temp_cyl = cyl.copy()

    # Create mesh to store the geometry of the cylinders, with one
    # entire cylinder per entry of 'vect'
    tm = mesh()
    # Create data arrays
    num_vertex = vect.shape[0]*cyl.vertex.shape[0]
    num_face = vect.shape[0]*cyl.face.shape[0]
    tm.vertex = np.zeros((num_vertex, 3))
    tm.vcolor = np.zeros((num_vertex, 3))
    tm.face = np.zeros((num_face, 3), dtype=np.int_)

    # Add a cylinder for each vector
    vertex_index = 0
    face_index = 0
    for i in range(vect.shape[0]):
        # Compute angle between vector and [0, 0, length]
        # General formula:
        #     angle = acos(dot(cyl_axis, vector)) / (|cyl_axis||vector|)
        # Specialized to [0, 0, length]:
        ang = math.acos(vect[i, 2]/np.linalg.norm(vect[i, :]))
        # Compute axis of rotation
        # General formula:
        #     axis = cross(cyl_axis, vector)
        # Specialized to [0, 0, length]:
        axs = np.array([-length*vect[i, 1], length*vect[i, 0], 0])
        # Rotate cylinder by angle and axis
        R = rotation_matrix(ang, axs)
        # Retrieve saved geometry of cylinder
        temp_cyl.vertex[:] = cyl.vertex[:]
        # Apply rotation matrix
        for j in range(temp_cyl.vertex.shape[0]):
            temp_cyl.vertex[j, :] = np.dot(R, temp_cyl.vertex[j, :].T)
        # Translate vector by adding its origin
        temp_cyl.vertex += pos[i, :]
        # Add geometry of cylinder to mesh
        tm.vertex[vertex_index:(vertex_index + temp_cyl.vertex.shape[0]), :] =\
            temp_cyl.vertex
        tm.vcolor[vertex_index:(vertex_index + temp_cyl.vertex.shape[0]), :] =\
            color
        tm.face[face_index:(face_index + temp_cyl.face.shape[0]), :] = \
            temp_cyl.face + vertex_index
        vertex_index += temp_cyl.vertex.shape[0]
        face_index += temp_cyl.face.shape[0]

    return tm


def create_lines(line, radius=0.03, color=[1, 0, 0]):
    """Create geometry to represent a set of lines

    Parameters
    ----------
    line : array_like
        An array of shape (n, 6), representing the lines. Line 'i' is
        defined by the points line(i, 0:3) and line(i, 3:6).
    radius : float
        Radius of the cylinder representing each line (the default
        value is 0.03)
    color : array_like
        Color of each line (the default value is red)

    Returns
    -------
    tm : mesh
        Mesh object representing the lines

    Notes
    -----
    The function creates a mesh composed of multiple small cylinders
    that represent a set of lines. Each cylinder connects the two end
    points of the corresponding line.
    """
 
    # Create a canonical cylinder with origin at [0, 0, 0] and
    # pointing to [0, 0, 1]
    cyl = create_cylinder(radius, 1, 8, 2, True)
    cyl.vertex += np.array([0.0, 0.0, 0.5])
    temp_cyl = cyl.copy()

    # Create mesh to store the geometry of the cylinders, with one
    # entire cylinder per entry of 'vect'
    tm = mesh()
    # Create data arrays
    num_vertex = line.shape[0]*cyl.vertex.shape[0]
    num_face = line.shape[0]*cyl.face.shape[0]
    tm.vertex = np.zeros((num_vertex, 3))
    tm.vcolor = np.zeros((num_vertex, 3))
    tm.face = np.zeros((num_face, 3), dtype=np.int_)

    # Add a cylinder for each line
    vertex_index = 0
    face_index = 0
    for i in range(line.shape[0]):
        # Create vector that captures the orientation of the line
        vect = line[i, 3:6] - line[i, 0:3]
        length = np.linalg.norm(vect) # Length will also be used below
        vect /= length
        # Compute angle between vector and the cylinder axis [0, 0, 1]
        # General formula:
        #     angle = acos(dot(axis, vector)) / (|axis||vector|)
        # Specialized to [0, 0, 1]:
        ang = math.acos(vect[2])
        # Compute axis of rotation
        # General formula:
        #     axis = cross(axis, vector)
        # Specialized to [0, 0, 1]:
        axs = np.array([-vect[1], vect[0], 0])
        # Rotate cylinder by angle and axis
        R = rotation_matrix(ang, axs)
        # Retrieve saved geometry of cylinder
        temp_cyl.vertex[:] = cyl.vertex[:]
        # Scale cylinder by length of line
        temp_cyl.vertex *= length
        # Apply rotation matrix
        for j in range(temp_cyl.vertex.shape[0]):
            temp_cyl.vertex[j, :] = np.dot(R, temp_cyl.vertex[j, :].T)
        # Add first endpoint to translate line
        temp_cyl.vertex += line[i, 0:3]
        # Add geometry of cylinder to mesh
        tm.vertex[vertex_index:(vertex_index + temp_cyl.vertex.shape[0]), :] =\
            temp_cyl.vertex
        tm.vcolor[vertex_index:(vertex_index + temp_cyl.vertex.shape[0]), :] =\
            color
        tm.face[face_index:(face_index + temp_cyl.face.shape[0]), :] = \
            temp_cyl.face + vertex_index
        vertex_index += temp_cyl.vertex.shape[0]
        face_index += temp_cyl.face.shape[0]

    return tm


# Create a set of samples taken from the surface of a sphere
def create_sphere_samples(n, radius=1.0, center=[0.0, 0.0, 0.0]):
    """Sample a set of points from the surface of a sphere

    Parameters
    ----------
    n : int
        Number of samples to compute
    radius : float
        Radius of the sphere (the default value is 1)
    center : array_like
        3D position with the center of the sphere (the default value is
        the origin)

    Returns
    -------
    pc : geomproc.pcloud
        Point cloud object storing the samples

    Notes
    -----
    The function creates a set of samples taken from the surface of the
    specified sphere. The sampling is performed to avoid bias at the
    poles of the sphere.

    See Also
    --------
    geomproc.creation.create_sphere

    Examples
    --------
    >>> import geomproc
    >>> tm = geomproc.create_sphere_samples(1000, 1.0)
    """

    # Initialize data
    point = np.zeros((n, 3))
    normal = np.zeros((n, 3))

    # Create n samples
    for i in range(n):
        # Get two random numbers
        u = random.random()
        v = random.random()

        # Use u to define the angle theta along one direction of the sphere
        theta = u * 2.0*math.pi
        # Use v to define the angle phi along the other direction of the sphere
        phi = math.acos(2.0*v - 1.0)

        # Define the normal and point based on theta and phi
        normal[i, :] = [math.cos(theta)*math.sin(phi), 
                        math.sin(theta)*math.sin(phi), math.cos(phi)]
        normal[i, :] /= np.linalg.norm(normal[i, :])
        point[i, :] = center + radius*normal[i, :]

    # Return output
    pc = pcloud()
    pc.point = point
    pc.normal = normal
    return pc
