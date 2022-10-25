#
# GeomProc: geometry processing library in python + numpy
#
# Copyright (c) 2008-2021 Oliver van Kaick <ovankaic@gmail.com>
# under the MIT License.
#
# See file LICENSE.txt for details on copyright licenses.
#
"""This module contains functions for loading geometric datasets of the
GeomProc geometry processing library.
"""


import numpy as np
import copy
import math
import random

from .mesh import *
from .pcloud import *


#### Functions for loading geometric datasets

# Create a model by loading its data from a file
def load(filename):
    """Load a geometric dataset from a file

    Parameters
    ----------
    filename : string
        Name of the input filename

    Returns
    -------
    dataset : object
        Loaded dataset, either a mesh or point cloud

    Notes
    -----
    The method reads all the information from the file and
    determines the type of dataset to be created, either a mesh with
    vertices and faces or a point cloud with only points.

    See Also
    --------
    geomproc.mesh
    geomproc.pcloud

    Examples
    --------
    >>> import geomproc
    >>> tm = geomproc.load('sphere.obj')
    """

    # Check the file extension and call the relevant method to load
    # the file
    part = filename.split('.')
    if part[-1].lower() == 'obj':
        return load_obj(filename)
    elif part[-1].lower() == 'off':
        return load_off(filename)
    else:
        raise RuntimeError('file format "'+part[-1]+'" not supported')


# Load a mesh or point cloud from a file in obj format
def load_obj(filename):

    # Parse a vertex description of the form <vid>/<tid>/<nid>
    def get_parts(st):
        # Split vertex description into parts
        arr = st.split('/')
        # Initialize all parts with default values
        vid = None
        tid = None
        nid = None
        # Parse vertex description based on number of parts
        if len(arr) == 3:
            vid = arr[0]
            tid = arr[1]
            nid = arr[2]
        elif len(arr) == 2:
            vid = arr[0]
            tid = arr[1]
        elif len(arr) == 1:
            vid = arr[0]
        # Convert data into numbers
        vid = int(vid)-1
        if tid != None:
            tid = int(tid)-1
        if nid != None:
            nid = int(nid)-1
        return vid, tid, nid

    # Temporary lists to store the mesh data
    # Basic mesh: faces and vertices
    vertex = []
    face = []
    # Vertex colors
    vcolor = []
    # Normals and texture coordinates that can be reused
    normal = []
    uv = []
    # Corner descriptions: references to normals and texture coordinates
    cnormal = []
    cuv = []
    # Open the file
    with open(filename, 'r') as f: 
        # Process each line of the file
        for line in f:
            # Remove extra whitespace at the beginning or end of line
            line = line.strip()
            # Ignore empty lines
            if len(line) < 1:
                continue
            # Ignore comments
            if line[0] == '#':
                continue
            # Parse line
            part = line.split(' ')
            # Vertex command
            if part[0] == 'v':
                if (len(part) != 4) and (len(part) != 7):
                    raise RuntimeError('v command should have 3 or 6 numbers')
                vertex.append([float(part[1]), float(part[2]), float(part[3])])
                if len(part) == 7:
                    vcolor.append([float(part[4]), float(part[5]), float(part[6])])
            # Vertex normal command
            if part[0] == 'vn':
                if len(part) < 4:
                    raise RuntimeError('vn command has less than 3 numbers')
                normal.append([float(part[1]), float(part[2]), float(part[3])])
            # Vertex texture coordinates command
            if part[0] == 'vt':
                if len(part) < 3:
                    raise RuntimeError('vn command has less than 2 numbers')
                uv.append([float(part[1]), float(part[2])])
            # Vertex color command
            if part[0] == 'vc':
                if len(part) < 3:
                    raise RuntimeError('vc command has less than 2 numbers')
                vcolor.append([float(part[1]), float(part[2]), float(part[3])])
            # Face command
            if part[0] == 'f':
                # Check consistency of command
                if (len(part) < 4) or (len(part) > 5):
                    raise RuntimeError('f command should have 3 or 4 groups of numbers')
                # Get first three ids of face
                [vid1, tid1, nid1] = get_parts(part[1])
                [vid2, tid2, nid2] = get_parts(part[2])
                [vid3, tid3, nid3] = get_parts(part[3])
                # Add face
                face.append([vid1, vid2, vid3])
                if nid1 != None:
                    cnormal.append([nid1, nid2, nid3])
                if tid1 != None:
                    cuv.append([tid1, tid2, tid3])
                # Check if it is a quad face
                if len(part) == 5:
                    # Parse last entry
                    [vid4, tid4, nid4] = get_parts(part[4])
                    # Add second face
                    face.append([vid1, vid3, vid4])
                    if nid1 != None:
                        cnormal.append([nid1, nid3, nid4])
                    if tid1 != None:
                        cuv.append([tid1, tid3, tid4])

    # Compose output object
    if len(face) > 0:
        # Create a mesh object
        output = mesh()
        # Vertices and faces
        output.vertex = np.array(vertex)
        output.face = np.array(face, dtype=np.int_)
        output.vcolor = np.array(vcolor)
        # Corner normals
        if len(cnormal) > 0:
            temp = []
            for ne in cnormal:
                temp_normal = [[normal[ne[0]][0], normal[ne[0]][1], normal[ne[0]][2]],
                               [normal[ne[1]][0], normal[ne[1]][1], normal[ne[1]][2]],
                               [normal[ne[2]][0], normal[ne[2]][1], normal[ne[2]][2]]]
                temp.append(temp_normal)
            output.cnormal = np.array(temp)
        # Corner texture coordinates
        if len(cuv) > 0:
            temp = []
            for te in cuv:
                temp_uv = [[uv[te[0]][0], uv[te[0]][1]],
                           [uv[te[1]][0], uv[te[1]][1]],
                           [uv[te[2]][0], uv[te[2]][1]]]
                temp.append(temp_uv)
            output.cuv = np.array(temp)
    else:
        # Create a point cloud object
        output = pcloud()
        # Points and colors
        output.point = np.array(vertex)
        output.color = np.array(vcolor)
        # Other properties
        # if len(vnormal) > 0:
            # if len(vnormal) == len(vertex):
                # output.normal = np.array(vnormal)
        if len(normal) > 0:
            if len(normal) == len(vertex):
                output.normal = np.array(normal)
    # Return output object
    return output


# Load a mesh or point cloud from a file in off format
def load_off(filename):
    # Temporary lists to store the mesh data
    # Basic mesh: faces and vertices
    vertex = []
    face = []
    # Open the file
    with open(filename, 'r') as f: 
        # Read header
        # Read file identifier
        line = f.readline()
        line = line.strip()
        if line[0:3] != 'OFF':
            raise RuntimeError("file does not have the 'OFF' identifier")
        # Read number of vertices and faces from header
        part = line.split()
        # Check if info is on the same line as the file identifier
        if len(part) == 4:
            # Discard file information from the parsed list
            part = [part[1], part[2], part[3]]
        else:
            # If info is not on the same line, try to read it from the
            # next line
            line = f.readline()
            line = line.strip()
            part = line.split()
            if len(part) != 3:
                raise RuntimeError('missing file information in the file header')
        # Get number of vertices and faces
        num_vertices = int(part[0])
        num_faces = int(part[1])
        # Read data
        # Read vertices
        for i in range(num_vertices):
            # Read line
            line = f.readline()
            line = line.strip()
            # Parse line
            part = line.split()
            if len(part) != 3:
                raise RuntimeError('vertex entry should have exactly 3 numbers')
            # Transform entries into numbers
            v1 = float(part[0])
            v2 = float(part[1])
            v3 = float(part[2])
            # Append data
            vertex.append([v1, v2, v3])
        # Read faces
        for i in range(num_faces):
            # Read line
            line = f.readline()
            line = line.strip()
            # Parse line
            part = line.split()
            if (len(part) != 4) and (len(part) != 5):
                raise RuntimeError('face entry should have 4 or 5 numbers')
            # Transform entries into numbers
            id1 = int(part[1])
            id2 = int(part[2])
            id3 = int(part[3])
            # Append face
            face.append([id1, id2, id3])
            # Check if it is a quad face
            if len(part) == 5:
                # Get remaining number
                id4 = int(part[4])
                # Add second face
                face.append([id1, id3, id4])

    # Compose output object
    if len(face) > 0:
        # Create mesh structure
        output = mesh()
        # Vertices and faces
        output.vertex = np.array(vertex)
        output.face = np.array(face, dtype=np.int_)
    else:
        # Create a point cloud object
        output = pcloud()
        # Points and colors
        output.point = np.array(vertex)

    # Return output object
    return output
