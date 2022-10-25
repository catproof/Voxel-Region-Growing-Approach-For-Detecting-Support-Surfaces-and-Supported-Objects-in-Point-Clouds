#
# GeomProc: geometry processing library in python + numpy
#
# Copyright (c) 2008-2021 Oliver van Kaick <ovankaic@gmail.com>
# under the MIT License.
#
# See file LICENSE.txt for details on copyright licenses.
#
"""This module contains the mesh class of the GeomProc geometry
processing library.
"""


import numpy as np
import copy
import math
import random

from .write_options import *
from .misc import *
from .pcloud import *
from .kdtree import *


# A basic triangle mesh data structure
class mesh:
    """A class that represents a triangle mesh

    Notes
    -----
    The class stores a triangle mesh represented by a list of vertices
    and a list of faces referencing the list of vertices. Additional
    attributes can be optionally stored.

    Attributes
    ----------
    vertex : numpy.array_like
        Vertices of the mesh. The array should be of shape (n, 3), where
        n is the number of vertices in the mesh. Each row of the array
        stores one vertex, and the columns of the array represent the x,
        y, and z coordinates of a vertex.

    face : numpy.array_like
        Faces of the mesh. The array should be of shape (m, 3), where m
        is the number of faces in the mesh. Each row of the array stores
        one face, and the columns of the array represent the vertex
        references of the faces. Vertex references are integer values
        starting at zero. Only triangular faces are supported.

    vnormal : numpy.array_like
        Vertex normals. The array should be either empty (to indicate
        that this attribute is not present) or of shape (n, 3), where n
        is the number of vertices in the mesh. The i-th row of the array
        stores the normal vector for the i-th vertex in the mesh, and
        the columns of the array are the x, y, and z components of the
        normal vector.

    vcolor : numpy.array_like
        Vertex colors. The array should be either empty (to indicate
        that this attribute is not present) or of shape (n, 3), where n
        is the number of vertices in the mesh. The i-th row of the array
        stores the RGB color for the i-th vertex in the mesh in the
        order r, g, and b.

    vuv : numpy.array_like
        Vertex texture coordinates (UVs). The array should be either
        empty (to indicate that this attribute is not present) or of
        shape (n, 2), where n is the number of vertices in the mesh. The
        i-th row of the array stores the 2D texture coordintes (u, v)
        for the i-th vertex in the mesh.

    fnormal : numpy.array_like
        Face normals. The array should be either empty (to indicate that
        this attribute is not present) or of shape (m, 3), where m is
        the number of faces in the mesh. The i-th row of the array
        stores the normal vector for the i-th face in the mesh, and the
        columns of the array are the x, y, and z components of the
        normal vector.

    cnormal : numpy.array_like
        Corner normals. This attribute stores the three normal vectors
        for the three corners of each face. The array should be either
        empty (to indicate that this attribute is not present) or of
        shape (m, 3, 3), where m is the number of faces in the mesh.
        The entry cnormal[i, j, :] stores the normal for the j-th
        corner of the i-th face in the mesh.

    cuv : numpy.array_like
        Corner texture coordinates (UVs). This attributes stores the
        three texture coordinates for the three corners of each face.
        The array should be either empty (to indicate that this
        attribute is not present) or of shape (m, 3, 3), where m is the
        number of faces in the mesh. The entry cuv[i, j, :] stores the
        2D texture coordinates (u, v) for the j-th corner of the i-th
        face in the mesh.

    vif : list of lists
        Connectivity information: list of vertices incident to a face.
        The i-th entry of this list stores the list of all faces
        incident to the i-th vertex in the mesh.

    fif : list of lists
        Connectivity information: list of faces neighboring a face.
        The i-th entry of this lists stores the list of all faces
        neighboring the i-th face of the mesh. Two faces are neighbors
        if they share an edge.

    viv : lists of lists
        Connectivity information: list of vertices neighboring a vertex.
        The i-th entry of this lists stores the list of all vertices
        neighboring the i-th vertex of the mesh. Two vertices are neighbors
        if they are connected by an edge.
    """

    def __init__(self):

        # Initialize all attributes

        # Mesh data
        # Vertex coordinates
        self.vertex = np.zeros((0, 3), dtype=np.single)
        # Triangular faces: integer indices starting at zero
        self.face = np.zeros((0, 3), dtype=np.int_)
        # Vertex normals
        self.vnormal = np.zeros((0, 3), dtype=np.single)
        # Vertex colors
        self.vcolor = np.zeros((0, 3), dtype=np.single)
        # Vertex texture coordinates
        self.vuv =  np.zeros((0, 2), dtype=np.single)
        # Face normals
        self.fnormal = np.zeros((0, 3), dtype=np.single)
        # Corner normals: one normal for each corner of a face
        self.cnormal = np.zeros((0, 0),  dtype=np.single)
        # Corner texture coordinates: one (u, v) pair for each corner of a face
        self.cuv = np.zeros((0, 0),  dtype=np.single)

        # Connectivity information
        # Vertex to faces references
        self.vif = []
        # Face to neighboring faces references
        self.fif = []
        # Vertex to neighboring vertices references
        self.viv = []

    # Data methods

    # Deep copy
    def copy(self):
        """Perform a deep copy of the mesh

        Parameters
        ----------
        None

        Returns
        -------
        tm : mesh
            New copied mesh
        """

        tm = mesh()
        tm.vertex = self.vertex.copy()
        tm.face = self.face.copy()
        tm.vnormal = self.vnormal.copy()
        tm.vcolor = self.vcolor.copy()
        tm.vuv = self.vuv.copy()
        tm.fnormal = self.fnormal.copy()
        tm.cnormal = self.cnormal.copy()
        tm.cuv = self.cuv.copy()
        tm.vif = copy.deepcopy(self.vif)
        tm.fif = copy.deepcopy(self.fif)
        tm.viv = copy.deepcopy(self.viv)
        return tm

    # Append one mesh to another
    def append(self, tm):
        """Append a mesh to the current mesh object

        Parameters
        ----------
        tm : geomproc.mesh
            Mesh to be appended to the current mesh

        Returns
        -------
        None
        """

        # Merge arrays if any of two are non-zero
        def merge_arrays(arr1, arr2, len1, len2, dim, dtype, default=0):
            if (arr1.shape[0] == 0) and (arr2.shape[0] == 0):
                new_arr = np.zeros((0, dim), dtype)
            elif (arr1.shape[0] > 0) and (arr2.shape[0] > 0):
                new_arr = np.concatenate((arr1, arr2), 0)
            elif (arr1.shape[0] > 0) and (arr2.shape[0] == 0):
                new_arr = np.zeros((arr1.shape[0] + len2, dim), dtype)
                new_arr[0:arr1.shape[0], :] = arr1
                new_arr[arr1.shape[0]:(arr1.shape[0] + len2), :] = default
            else:
                new_arr = np.zeros((len1 + arr2.shape[0], dim), dtype)
                new_arr[0:len1, :] = default
                new_arr[len1:(len1 + arr2.shape[0]), :] = arr2
            return new_arr

        def merge_2d_arrays(arr1, arr2, len1, len2, dim1, dim2, dtype, default=0):
            if (arr1.shape[0] == 0) and (arr2.shape[0] == 0):
                new_arr = np.zeros((0, dim1, dim2), dtype)
            elif (arr1.shape[0] > 0) and (arr2.shape[0] > 0):
                new_arr = np.concatenate((arr1, arr2), 0)
            elif (arr1.shape[0] > 0) and (arr2.shape[0] == 0):
                new_arr = np.zeros((arr1.shape[0] + len2, dim1, dim2), dtype)
                new_arr[0:arr1.shape[0], :] = arr1
                new_arr[arr1.shape[0]:(arr1.shape[0] + len2), :] = default
            else:
                new_arr = np.zeros((len1 + arr2.shape[0], dim1, dim2), dtype)
                new_arr[0:len1, :] = default
                new_arr[len1:(len1 + arr2.shape[0]), :] = arr2
            return new_arr


        # Save previous dimensions
        svc = self.vertex.shape[0]
        tvc = tm.vertex.shape[0]
        sfc = self.face.shape[0]
        tfc = tm.face.shape[0]

        # Combine main arrays
        self.vertex = np.concatenate((self.vertex, tm.vertex), 0)
        self.face = np.concatenate((self.face, tm.face), 0)
        # Update vertex references
        self.face[sfc:(sfc + tfc)] += svc

        # Combine all secondary data arrays
        self.vnormal = merge_arrays(self.vnormal, tm.vnormal, svc, tvc, 3, np.single)
        self.vcolor = merge_arrays(self.vcolor, tm.vcolor, svc, tvc, 3, np.single, 0.8)
        self.vuv = merge_arrays(self.vuv, tm.vuv, svc, tvc,  2, np.single)
        self.fnormal = merge_arrays(self.fnormal, tm.fnormal, sfc, tfc,  3, np.single)
        self.cnormal = merge_2d_arrays(self.cnormal, tm.cnormal, sfc, tfc, 3, 3, np.single)
        self.cuv = merge_2d_arrays(self.cnormal, tm.cnormal, sfc, tfc, 3, 3, np.single)

        # Recompute connectivity information if it was present in either
        # of the two meshes
        if (len(self.vif) > 0) or (len(tm.vif) > 0):
            self.compute_connecivity()

    # I/O methods

    # Save a mesh to a file
    def save(self, filename, wo = write_options()): 
        """Save a mesh to a file

        Parameters
        ----------
        filename : string
            Name of the output filename
        wo : write_options object, optional
            Object with flags that indicate which fields of the mesh
            should be written to the output file

        Returns
        -------
        None

        Notes
        -----
        The method saves the mesh information into a file. The file
        format is determined from the filename extension. Currently, the
        obj and off file formats are supported. By default, only
        vertices and faces are written into the file. Other information
        is written if the corresponding flags are set in the
        write_options object. Not all flags are supported by all file
        formats.

        See Also
        --------
        geomproc.write_options

        Examples
        --------
        >>> import geomproc
        >>> tm = geomproc.create_sphere(1.0, 30, 30)
        >>> tm.save('sphere.obj')
        """

        # Check the file extension and call the relevant method to save
        # the file
        part = filename.split('.')
        if part[-1].lower() == 'obj':
            return self.save_obj(filename, wo)
        elif part[-1].lower() == 'off':
            return self.save_off(filename, wo)
        else:
            raise RuntimeError('file format "'+part[-1]+'" not supported')

    # Save a mesh to a file in obj format
    def save_obj(self, filename, wo = write_options()):
        # Check consistency of write options
        if ((wo.write_vertex_normals and wo.write_face_normals) or
            (wo.write_vertex_normals and wo.write_corner_normals) or
            (wo.write_face_normals and wo.write_corner_normals)):
            raise RuntimeError('cannot specify multiple write_<xx>_normals options at the same time')
        if wo.write_vertex_uvs and wo.write_corner_uvs:
            raise RuntimeError('cannot specify multiple write_<xx>_uvs options at the same time')

        # Create material file, if needed
        if (wo.write_vertex_uvs or wo.write_corner_uvs) and (wo.texture_name != ''):
            # Write material file with texture name
            fn_part = filename.split('.')
            material_filename = fn_part[0] + '.mtl'
            with open(material_filename, 'w') as f: 
                f.write('newmtl textured\n')
                f.write('Ka 1.000 1.000 1.000\n')
                f.write('Kd 1.000 1.000 1.000\n')
                f.write('Ks 0.000 0.000 0.000\n')
                f.write('Ns 10.0\n')
                f.write('d 1.0\n')
                f.write('illum 0\n')
                f.write('map_Kd ' + wo.texture_name + '\n')

        # Open the file
        with open(filename, 'w') as f: 
            # Material file reference
            if (wo.write_vertex_uvs or wo.write_corner_uvs) and (wo.texture_name != ''):
                # Request obj file to use material
                f.write('mtllib ' + material_filename + '\n')
                f.write('usemtl textured\n')

            # Write vertices
            if wo.write_vertex_colors:
                for i in range(self.vertex.shape[0]):
                    f.write('v '+str(self.vertex[i, 0])+' '+
                            str(self.vertex[i, 1])+' '+
                            str(self.vertex[i, 2])+' '+
                            str(self.vcolor[i, 0])+' '+
                            str(self.vcolor[i, 1])+' '+
                            str(self.vcolor[i, 2])+'\n')
            else:
                for i in range(self.vertex.shape[0]):
                    f.write('v '+str(self.vertex[i, 0])+' '+
                            str(self.vertex[i, 1])+' '+
                            str(self.vertex[i, 2])+'\n')

            # Write list of texture coordinates
            if wo.write_vertex_uvs:
                for i in range(self.vuv.shape[0]):
                    f.write('vt '+str(self.vuv[i, 0])+' '+
                            str(self.vuv[i, 1])+'\n')
            elif wo.write_corner_uvs:
                for i in range(self.cuv.shape[0]):
                    for j in range(self.cuv.shape[1]):
                        f.write('vt '+str(self.cuv[i, j, 0])+' '+
                                str(self.cuv[i, j, 1])+'\n')

            # Write list of normals
            if wo.write_vertex_normals:
                for i in range(self.vnormal.shape[0]):
                    f.write('vn '+str(self.vnormal[i, 0])+' '+
                            str(self.vnormal[i, 1])+' '+
                            str(self.vnormal[i, 2])+'\n')
            elif wo.write_face_normals:
                for i in range(self.fnormal.shape[0]):
                    f.write('vn '+str(self.fnormal[i, 0])+' '+
                            str(self.fnormal[i, 1])+' '+
                            str(self.fnormal[i, 2])+'\n')
            elif wo.write_corner_normals:
                for i in range(self.cnormal.shape[0]):
                    for j in range(self.cnormal.shape[1]):
                        f.write('vn '+str(self.cnormal[i, j, 0])+' '+
                                str(self.cnormal[i, j, 1])+' '+
                                str(self.cnormal[i, j, 2])+'\n')

            # Write faces
            if ((not wo.write_vertex_normals) and 
                (not wo.write_face_normals) and 
                (not wo.write_corner_normals) and
                (not wo.write_vertex_uvs) and 
                (not wo.write_corner_uvs)):
                # Write only vertex indices
                for i in range(self.face.shape[0]):
                    f.write('f '+str(self.face[i, 0]+1)+' '+
                            str(self.face[i, 1]+1)+' '+
                            str(self.face[i, 2]+1)+'\n')
            else:
                # Write more attributes
                # Check which attribute separators we need to write
                sep1 = '/'
                sep2 = '/'
                if ((wo.write_vertex_uvs or wo.write_corner_uvs) and
                    (not (wo.write_vertex_normals or wo.write_face_normals or 
                     wo.write_corner_normals))):
                    sep2 = ''
                # Write faces
                for i in range(self.face.shape[0]):
                    # Get face attributes
                    # Vertex ids
                    v0 = str(self.face[i, 0]+1)
                    v1 = str(self.face[i, 1]+1)
                    v2 = str(self.face[i, 2]+1)
                    # Texture coordinates
                    if wo.write_vertex_uvs:
                        t0 = v0
                        t1 = v1
                        t2 = v2
                    elif wo.write_corner_uvs:
                        t0 = str(i*3 + 0 + 1) # Indexing starts at 1, so + 1
                        t1 = str(i*3 + 1 + 1)
                        t2 = str(i*3 + 2 + 1)
                    else:
                        t0 = ''
                        t1 = ''
                        t2 = ''
                    # Normals
                    if wo.write_vertex_normals:
                        n0 = v0
                        n1 = v1
                        n2 = v2
                    elif wo.write_face_normals:
                        n0 = str(i + 1)
                        n1 = str(i + 1)
                        n2 = str(i + 1)
                    elif wo.write_corner_normals:
                        n0 = str(i*3 + 0 + 1)
                        n1 = str(i*3 + 1 + 1)
                        n2 = str(i*3 + 2 + 1)
                    else:
                        n0 = ''
                        n1 = ''
                        n2 = ''
                    # Write the face with all the attributes
                    f.write('f ' + v0 + sep1 + t0 + sep2 + n0 + ' ' + v1 + sep1 + t1 + sep2 + n1 + ' ' + v2 + sep1 + t2 + sep2 + n2 + '\n')

    # Save a mesh to a file in off format
    def save_off(self, filename, wo = write_options()):
        # Open the file
        with open(filename, 'w') as f: 
            # Write header
            f.write('OFF\n')
            f.write(str(self.vertex.shape[0])+' '+str(self.face.shape[0])+' 0\n')
            # Write data
            # Write vertices
            for i in range(self.vertex.shape[0]):
                f.write(str(self.vertex[i, 0])+' '+
                        str(self.vertex[i, 1])+' '+
                        str(self.vertex[i, 2])+'\n')
            # Write faces
            for i in range(self.face.shape[0]):
                f.write('3 '+str(self.face[i, 0])+' '+
                             str(self.face[i, 1])+' '+
                             str(self.face[i, 2])+'\n')

    # Geometry methods

    # Normalize the vertex coordinates of a mesh into a cube with
    # coordinates (target_min, target_max) along each dimension, while
    # preserving the aspect ratio of the shape
    def normalize(self, target_min = -1, target_max = 1):
        """Normalize the coordinates of a mesh

        Parameters
        ----------
        target_min : float, optional
            Target minimum coordinate value for the mesh vertices
        target_max : float, optional
            Target maximum coordinate value for the mesh vertices

        Returns
        -------
        None

        Notes
        -----
        The method modifies the vertex positions so that the longest
        axis is mapped to the range [target_min, target_max] (which is
        [-1.0, 1.0] by default), while the other axes are mapped so as
        to preserve the aspect ratio of the model.

        See Also
        --------
        geomproc.mesh.mesh.vertex

        Examples
        --------
        >>> import geomproc
        >>> tm = geomproc.create_sphere(1.0, 30, 30)
        >>> tm.normalize()
        """

        # For each coordinate x, y, and z, we obtain a range of values to
        # map, where the range for coordinate i is given by range_i = max_i
        # - min_i, where max_i is the maximum value for coordinate i for
        # all points, and min_i is defined analogously. Then, we would like
        # to map the coordinate with the largest range range_max to
        # (target_min, target_max), and map the other coordinates i by
        # keeping the proportion of range_i to range_max, so that the
        # aspect ratio of the shape is maintained.
        #
        # Thus, we would like to map a specific range_i from (max_i -
        # min_i) to (target_min*p, target_max*p), where p is the proportion
        # of range_i to the largest range, given by range_i/range_max. This
        # mapping is illustrated with the following diagram, where r is the
        # resulting mapped value:
        #
        # min_i     target_min*p        
        #   |        |
        # coord_i    r         
        #   |        |
        # max_i      target_max*p       
        #
        # This mapping problem gives the following ratio equality:
        #
        # coord_i - min_i     r -target_min*p
        # ---------------  =  ---------------------------
        # max_i - min_i       (target_max - target_min)*p
        #
        # By multiplying the fractions, and plugging in the definition of
        # p, we get the equation used in the code:
        #
        #     coord_i*(target_max - target_min) -min_i*target_max + max_i*target_min
        # r = ----------------------------------------------------------------------
        #                                 (max - min)

        # Get min and max coordinates of all vertices
        min_pos = self.vertex.min(axis=0)
        max_pos = self.vertex.max(axis=0)
        # Compute constants for normalization mapping
        diff = max_pos - min_pos
        rng = diff.max()
        mult_const = target_max - target_min
        add_const = -min_pos*target_max + max_pos*target_min
        # Map vertex coordinates to new range
        self.vertex = (self.vertex*mult_const + add_const)/rng

    # Compute vertex and face normals
    def compute_vertex_and_face_normals(self):
        """Compute the normal vectors of vertices and faces in the mesh

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        The method sets up the vnormal and fnormal attributes of the
        mesh with the normal vectors of vertices and faces,
        respectively. It also sets up a temporary area attribute with
        the triangle areas.

        See Also
        --------
        geomproc.mesh.mesh.vnormal
        geomproc.mesh.mesh.fnormal

        Examples
        --------
        >>> import geomproc
        >>> tm = geomproc.create_sphere(1.0, 30, 30)
        >>> tm.compute_vertex_and_face_normals()
        """

        # Initialize all arrays
        self.vnormal = np.zeros((self.vertex.shape[0], 3))
        self.fnormal = np.zeros((self.face.shape[0], 3))
        self.farea = np.zeros((self.face.shape[0], 1))
        # Compute normals
        for i in range(self.face.shape[0]):
            # Get three vertices of face
            v = self.vertex[self.face[i, :], :]

            # Compute face normal and area
            vec0 = v[1, :] - v[0, :]
            vec1 = v[2, :] - v[0, :]
            nrm = np.cross(vec0, vec1)
            lng = np.linalg.norm(nrm)
            area = lng/2.0
            nrm = nrm / lng

            # Assign face normal and area
            self.fnormal[i, :] = nrm
            self.farea[i, :] = area

            # Add face normal to normal of vertices
            self.vnormal[self.face[i, 0], :] += area*nrm
            self.vnormal[self.face[i, 1], :] += area*nrm
            self.vnormal[self.face[i, 2], :] += area*nrm

        # Normalize normal vectors of vertices
        for i in range(self.vertex.shape[0]):
            self.vnormal[i, :] /= np.linalg.norm(self.vnormal[i, :])

    # Connectivity methods

    # Compute only vif list
    def compute_vif(self):
        """Compute vif mesh connectivity information

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        The method sets up the vif attribute of the mesh with
        information on connections between vertices and faces.

        See Also
        --------
        geomproc.mesh.mesh.vif

        Examples
        --------
        >>> import geomproc
        >>> tm = geomproc.create_sphere(1.0, 30, 30)
        >>> tm.compute_vif()
        """

        # Initialize all reference lists
        self.vif = [[] for i in range(self.vertex.shape[0])]
        # Populate reference lists
        for i in range(self.face.shape[0]):
            for j in range(3):
                self.vif[self.face[i, j]].append(i)

    # Compute all connectivity lists
    def compute_connectivity(self):
        """Compute mesh connectivity information

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        The method sets up the vif, fif, and viv attributes of the mesh
        with information on connections between vertices and faces.

        See Also
        --------
        geomproc.mesh.mesh.vif
        geomproc.mesh.mesh.fif
        geomproc.mesh.mesh.viv

        Examples
        --------
        >>> import geomproc
        >>> tm = geomproc.create_sphere(1.0, 30, 30)
        >>> tm.compute_connectivity()
        """

        # Compute vif first
        self.compute_vif()
        # Compute fif
        self.fif = [[] for i in range(self.face.shape[0])]
        for i in range(self.face.shape[0]):
            for j in range(3):
                for f in self.vif[self.face[i, j]]:
                    if ((f in self.vif[self.face[i, (j + 1) % 3]]) or \
                        (f in self.vif[self.face[i, (j + 2) % 3]])):
                        if (not (f in self.fif[i])) and (f != i):
                            self.fif[i].append(f)
        # Compute viv
        self.viv = [[] for i in range(self.vertex.shape[0])]
        for i in range(self.vertex.shape[0]):
            for f in self.vif[i]:
                for k in self.face[f, :]:
                    if (not (k in self.viv[i])) and (k != i):
                        self.viv[i].append(k)

    # Curvature
    def compute_curvature(self):
        """Compute curvatures of mesh vertices using the angles of triangles

        Parameters
        ----------
        None

        Returns
        -------
        negative : int, optional
            Negative is set to 1 if any negative cotangent weights were
            computed for the mesh. Otherwise, it is zero.

        Notes
        -----
        The method computes the curvature of mesh vertices and stores
        this information in the 'curv' attribute of the class. The entry
        curv[i, 0] is the minimal curvature for vertex 'i', curv[i, 1]
        is the maximal curvature, curv[i, 2] is the mean curvature,
        curv[i, 3] is the Gaussian curvature, and curv[i, 4] is the area
        of the Voronoi region around the vertex.

        Examples
        --------
        >>> import geomproc
        >>> tm = geomproc.create_sphere(1.0, 30, 30)
        >>> neg = tm.compute_curvature()
        """

        # The following method computes the curvature of mesh vertices. The
        # method is one of the simplest available, based on the angles of
        # the faces incident to the vertex. The code computes the curvatures
        # by iterating over the mesh faces, in contrast to traditional
        # implementations that iterate over the edges in the one-ring of
        # vertices.
        #
        # Reference for formulas: Section 2 in:
        # "Curvature Estimation for Unstructured Triangulations of Surfaces"
        # Rao V. Garimella and Blair K. Swartz
        # Technical Report LA-UR-03-8240
        # Los Alamos National Laboratory
        #
        # The original reference is:
        # "Generalizing barycentric coordinates for irregular n-gons"
        # M. Meyer, H. Lee, M. Desbrun, and A. H. Barr
        # Journal of Graphics Tools, 2002

        # Submethod
        #
        # Compute squared norm of a 3D vector
        def sq_norm(v):
            return v[0]*v[0] + v[1]*v[1] + v[2]*v[2]

        # Submethod
        #
        # Compute mean curvature for edge (v1i, v2i) and corner v3i according to
        # the geometric Laplacian
        # Also return the portion of the area of the Voronoi region around the
        # vertex corresponding to the edge in 'varea'
        #
        def compute_mean_curvature(v1i, v2i, v3i, negative):

            # Get vertex positions
            v1 = self.vertex[v1i, :]
            v2 = self.vertex[v2i, :]
            v3 = self.vertex[v3i, :]

            # Compute the area of the Voronoi region around v1 restricted to
            # the face
         
            # Compute the cotangents of the two corners opposite to v1
            vec1 = v2 - v3
            vec1 /= np.linalg.norm(vec1)
            vec2 = v2 - v1
            vec2 /= np.linalg.norm(vec2)
            cosine = np.dot(vec1, vec2)
            sine = np.linalg.norm(np.cross(vec1, vec2))
            cot_v2 = cosine/sine

            vec1 = v3 - v1
            vec1 /= np.linalg.norm(vec1)
            vec2 = v3 - v2
            vec2 /= np.linalg.norm(vec2)
            cosine = np.dot(vec1, vec2)
            sine = np.linalg.norm(np.cross(vec1, vec2))
            cot_v3 = cosine/sine

            # Compute the area based on cotangents and edge lengths
            #
            # Note that we could make the code more efficient by moving the
            # (1/8) factors to the normalization at the end

            # Voronoi area that touches (v1, v2)
            area_v2 = (1/8)*sq_norm(v1 - v2)*cot_v3
            # Voronoi area that touches (v1, v3)
            area_v3 = (1/8)*sq_norm(v1 - v3)*cot_v2
            # Total Voronoi area on this face for v1
            varea = area_v2 + area_v3 

            # Compute the mean curvature portion for this face
            H = area_v2*2*np.dot(v1 - v2, self.vnormal[v1i, :])/sq_norm(v1 - v2) + \
                area_v3*2*np.dot(v1 - v3, self.vnormal[v1i, :])/sq_norm(v1 - v3) 

            # Check negativity of cotan weights
            if (cot_v2 < 0) or (cot_v3 < 0):
                negative = 1

            return [H, varea, negative]

        # Submethod
        #
        # Compute Gaussian curvature for vertex 'vindex' based on area of
        # Voronoi region around the vertex 'varea'
        def compute_Gaussian_curvature(v1i, v2i, v3i):

            # Get vertex positions
            v1 = self.vertex[v1i, :]
            v2 = self.vertex[v2i, :]
            v3 = self.vertex[v3i, :]

            # Compute angle for Gaussian curvature estimate
            #
            # Get two vectors linking the current vertex to the opposed
            # vertices, and normalize the vectors
            vec1 = v2 - v1
            vec1 /= np.linalg.norm(vec1)
            vec2 = v3 - v1
            vec2 /= np.linalg.norm(vec2)
            # Compute angle of the face at the current vertex using the
            # two vectors
            cosine = np.dot(vec1, vec2)
            if cosine > 1.0:
                cosine = 1.0
            elif cosine < -1.0:
                cosine = -1.0
            ang = math.acos(cosine)

            # Return angle as the estimate of Gaussian curvature at the vertex
            # restricted to this face
            return ang

        # Init curvature information
        self.curv = np.zeros((self.vertex.shape[0], 5))

        # Init flags
        negative = 0

        # Check if vertex normals are available
        # If not, compute them
        if self.vnormal.shape[0] == 0:
            self.compute_vertex_and_face_normals()

        # Compute mean curvature by iterating over faces
        for i in range(self.face.shape[0]):

            # Consider each vertex of the face
            for j in range(3):

                # Compute mean curvature and Voronoi area for vertex j
                # restricted at face i
                v1i = self.face[i, j]
                v2i = self.face[i, ((j+1) % 3)]
                v3i = self.face[i, ((j+2) % 3)]
                [H, varea, negative] = compute_mean_curvature(v1i, v2i, v3i, negative)

                # Add the partial curvature and partial Voronoi area to the
                # values computed so far
                self.curv[v1i, 2] += H
                self.curv[v1i, 4] += varea

        # Finalize estimate of mean curvature
        for i in range (self.vertex.shape[0]):
            # Normalize sum of mean curvature by Voronoi area around vertex
            self.curv[i, 2] /= self.curv[i, 4]

        # Compute Gaussian curvature by iterating over faces
        for i in range(self.face.shape[0]):

            # Consider each vertex of the face
            for j in range(3):

                # Compute Gaussian curvature for vertex j restricted at face i
                v1i = self.face[i, j]
                v2i = self.face[i, ((j+1) % 3)]
                v3i = self.face[i, ((j+2) % 3)]
                K = compute_Gaussian_curvature(v1i, v2i, v3i)

                # Add the partial curvature to the values computed so far
                self.curv[v1i, 3] += K

        # Finalize estimate of Gaussian curvature
        for i in range(self.vertex.shape[0]):
            # Normalize sum of mean curvature by Voronoi area around vertex
            varea = self.curv[i, 4]
            self.curv[i, 3] = (1/varea)*(2*math.pi - self.curv[i, 3])

        # Compute min and max curvature based on K and G already computed
        for i in range(self.vertex.shape[0]):
            # Retrieve mean and Gaussian curvatures
            H = self.curv[i, 2]
            K = self.curv[i, 3]
            # Compute min and max
            coeff = H*H - K
            if coeff < 0:
                sq = 0
            else:
                sq = math.sqrt(coeff)
            self.curv[i, 0] = H - sq #- sqrt(H^2 - K)
            self.curv[i, 1] = H + sq #+ sqrt(H^2 - K)
            # Find which is the minimal curvature and which one is the maximal
            if self.curv[i, 0] > self.curv[i, 1]:
                temp = self.curv[i, 0]
                self.curv[i, 0] = new_model.curv[i, 1]
                self.curv[i, 1] = temp

        return negative

    def data_to_color(self, data, invert=False, percent=0.01, minimum=1.0, maximum=-1.0):
        """Transform vertex data into vertex colors

        Parameters
        ----------
        data : array_like
            Data array to be mapped to colors. This should be an array
            of shape (n, 1), where n is the number of vertices in the
            mesh
        invert : boolean, optional
            Flag indicating whether the color map should be inverted or
            not. The default value is False
        percent : float, optional
            Percentage of values to discard at each end of the spectrum
            of data values, to compute more robust minimum and maximum
            values for the color mapping, ignoring extreme outliers. The
            default value is 0.01. To ignore robust statistics, set this
            parameter to zero
        minimum : float, optional
            Minimum to be used for defining the mapping. If the minimum
            is specified, then both 'minimum' and 'maximum' need to be
            specified and the 'percent' parameter is ignored. Otherwise
            the minimum and maximum are computed automatically from the
            data based on the 'percent' parameter and returned by the
            method. The explicit parameters 'minimum' and 'maximum' are
            useful if multiple mappings with the same scale need to be
            produced
        maximum : float, optional
            Maximum to be used for defining the mapping

        Returns
        -------
        minimum : float
            Minimum value that was used to compute the mapping
        maximum : float
            Maximum value that was used to compute the mapping

        See Also
        --------
        geomproc.mesh.mesh.data_to_color_with_zero
 
        Notes
        -----
        The method maps the values of a data array into colors and
        stores the colors in the 'vcolor' attribute of the mesh, so that
        each vertex has an associated color. The data values are mapped
        from [minimum, maximum] to the hue [0, 2/3] in the HSV color
        system and then transformed into RGB colors, so that the minimum
        value is red and the maximum value is blue. If 'invert' is True,
        the minimum value is blue and the maximum is red.
        """

        # Map data values to colors

        # Calculate min and max with percentages to obtain a more robust
        # mapping
        if minimum > maximum:
            val = np.sort(data)
            minimum_index = math.floor(percent*val.shape[0])
            minimum = val[minimum_index]

            maximum_index = math.ceil((1 - percent)*val.shape[0])
            if maximum_index > (val.shape[0]-1):
                maximum_index = val.shape[0]-1
            maximum = val[maximum_index]

        # Check inversion of mapping
        if invert:
            low = 2/3
            high = 0
        else:
            low = 0
            high = 2/3

        # Perform color mapping
        if minimum == maximum:
            # Avoid division by zero if min and max are identical
            if minimum < 0:
                result = low*np.ones(data.shape[0])
            else:
                result = high*np.ones(data.shape[0])
        else:
            result = map_val(data, low, high, minimum, maximum)

        # Get RGB colors
        # Saturation and brightness are set to 0.8
        self.vcolor = np.zeros((self.vertex.shape[0], 3))
        for i in range(self.vertex.shape[0]):
            self.vcolor[i, :] = hsv2rgb([result[i], 0.8, 0.8])

        return [minimum, maximum]

    def data_to_color_with_zero(self, data, invert=False, percent=0.01, minimum=1.0, maximum=-1.0):
        """Transform vertex data into vertex colors while preserving the
        zero crossing

        Parameters
        ----------
        data : array_like
            Data array to be mapped to colors. This should be an array
            of shape (n, 1), where n is the number of vertices in the
            mesh
        invert : boolean, optional
            Flag indicating whether the color map should be inverted or
            not. The default value is False
        percent : float, optional
            Percentage of values to discard at each end of the spectrum
            of data values, to compute more robust minimum and maximum
            values for the color mapping, ignoring extreme outliers. The
            default value is 0.01. To ignore robust statistics, set this
            parameter to zero
        minimum : float, optional
            Minimum to be used for defining the mapping. If the minimum
            is specified, then both 'minimum' and 'maximum' need to be
            specified and the 'percent' parameter is ignored. Otherwise
            the minimum and maximum are computed automatically from the
            data based on the 'percent' parameter and returned by the
            method. The explicit parameters 'minimum' and 'maximum' are
            useful if multiple mappings with the same scale need to be
            produced
        maximum : float, optional
            Maximum to be used for defining the mapping

        Returns
        -------
        minimum : float
            Minimum value that was used to compute the mapping
        maximum : float
            Maximum value that was used to compute the mapping

        Notes
        -----
        The method maps the values of a data array into colors and
        stores the colors in the 'vcolor' attribute of the mesh, so that
        each vertex has an associated color. The data values are mapped
        from [min, 0] to the hue [0, 1/3] in the HSV color system, and
        from [0, max] to the hue [1/3, 2/3], so that min is red, 0 is
        green, and max is blue. If 'invert' is True, min is blue, 0 is
        green, and max is red.

        See Also
        --------
        geomproc.mesh.mesh.data_to_color

        Examples
        --------
        >>> import geomproc
        >>> tm = geomproc.create_torus(2, 1, 30, 30)
        >>> neg = tm.compute_curvature()
        >>> [mn, mx] = tm.data_to_color(tm.curv[:, 3])
        >>> wo = geomproc.write_options()
        >>> wo.write_vertex_colors = True
        >>> tm.save('colored_torus.obj', wo)
        """

        # Map data values to colors

        # Calculate min and max with percentages to obtain a more robust
        # mapping
        if minimum > maximum:
            val = np.sort(data)
            minimum_index = math.floor(percent*val.shape[0])
            minimum = val[minimum_index]

            maximum_index = math.ceil((1 - percent)*val.shape[0])
            if maximum_index > (val.shape[0]-1):
                maximum_index = val.shape[0]-1
            maximum = val[maximum_index]

        # Check inversion of mapping
        if invert:
            low = 2/3
            middle = 1/3
            high = 0
        else:
            low = 0
            middle = 1/3
            high = 2/3

        # Perform color mapping
        if minimum == maximum:
            # Avoid division by zero if min and max are identical
            if abs(minimum) < np.finfo(float).eps:
                result = middle*np.ones(data.shape[0])
            elif minimum < 0:
                result = low*np.ones(data.shape[0])
            else:
                result = high*np.ones(data.shape[0])
        else:
            # Map positive and negative values independently
            result = data.copy()
            neg_index = np.where(result < 0)[0]
            pos_index = np.where(result >= 0)[0]
            if len(neg_index) > 0:
                if abs(minimum) < np.finfo(float).eps:
                    # minimum is zero
                    result[neg_index] = middle*np.ones(neg_index.shape[0])
                else:
                    result[neg_index] = map_val(result[neg_index], low, middle, minimum, 0)

            if len(pos_index) > 0:
                if abs(maximum) < np.finfo(float).eps:
                    # maximum is zero
                    result[pos_index] = middle*np.ones(pos_index.shape[0])
                else:
                    result[pos_index] = map_val(result[pos_index], middle, high, 0, maximum)

        # Get RGB colors
        # Saturation and brightness are set to 0.8
        self.vcolor = np.zeros((self.vertex.shape[0], 3))
        for i in range(self.vertex.shape[0]):
            self.vcolor[i, :] = hsv2rgb([result[i], 0.8, 0.8])

        return [minimum, maximum]

    def data_face_to_vertex(self, fdata):
        """Transform a face data array to a vertex data array

        Parameters
        ----------
        fdata : array_like
            Face data array. This should be an array of shape (m, 1),
            where m is the number of faces in the mesh

        Returns
        -------
        vdata : array_like
            Vertex data array. This should be an array of shape (n, 1),
            where n is the number of vertices in the mesh

        Notes
        -----
        The method takes data values defined per face of the mesh and
        transforms them into data values defined per vertex of the mesh.
        This is accomplished by averaging the data values for all the
        faces connected to a vertex and assigning the average to the
        vertex.

        See Also
        --------
        geomproc.mesh.mesh.data_to_color
        """

        # Check if dimensions of the input data are correct
        if fdata.shape[0] != self.face.shape[0]:
            raise RuntimeError('length of data array is different from the number of faces in the mesh')

        # Initialize vertex data array
        vdata = np.zeros(self.vertex.shape[0])
        # Initialize count array for computing an average
        count = np.zeros(self.vertex.shape[0])
        # Average data values at faces to define values at vertices
        for i in range(self.face.shape[0]):
            vdata[self.face[i, :]] += fdata[i]
            count[self.face[i, :]] += 1.0
        for i in range(self.vertex.shape[0]):
            vdata[i] /= count[i]
        # Return vertex data array
        return vdata

    # Sample points from a mesh
    def sample(self, num_samples):
        """Sample points uniformly across a mesh

        Parameters
        ----------
        num_samples : int
            Number of samples to collect

        Returns
        -------
        pc : pcloud
            A point cloud with the point samples and their normals

        Notes
        -----
        The methods uses face areas and normals to perform the sampling 

        See Also
        --------
        geomproc.pcloud

        Examples
        --------
        >>> import geomproc
        >>> tm = geomproc.create_sphere(1.0, 30, 30)
        >>> pc = tm.sample(100)
        """

        # Check if face normals and areas are available
        # If not, compute them
        if (self.fnormal.shape[0] == 0) or (self.farea.shape[0] == 0):
            self.compute_vertex_and_face_normals()

        # Initialize output
        pc = pcloud()
        pc.point = np.zeros((num_samples, 3))
        pc.normal = np.zeros((num_samples, 3))

        # Create cumulative sum based on triangle areas
        # Transform areas into a probability distribution
        norm_area = self.farea / self.farea.sum()
        # Compute cumulative sum of probability distribution
        cs = np.cumsum(norm_area)

        # Sample random points
        for i in range(num_samples):
            # First, randomly sample a triangle, with a probability
            # given by its areas
            # Get a random number
            r = random.random()
            # Find location in the cumulative sum where entry is larger than the
            # random number
            tri = np.where(cs > r)
            tri = tri[0][0]
            # Now, sample a random point in this triangle
            pc.point[i, :] = random_triangle_sample(self, tri)
            # Copy normal vector
            pc.normal[i, :] = self.fnormal[tri, :]

        return pc

    # Create uniform Laplacian matrix
    def uniform_laplacian(self):
        """Create the uniform Laplacian operator for the mesh

        Parameters
        ----------
        None

        Returns
        -------
        L : numpy.array_like
            Laplacian matrix of dimensions n x n, where n is the number
            of vertices in the mesh

        Notes
        -----
        The method constructors the discrete uniform Laplacian operator
        based on the geometry of the mesh.

        See Also
        --------
        geomproc.mesh.mesh.geometric_laplacian

        Examples
        --------
        >>> import geomproc
        >>> tm = geomproc.create_sphere(1.0, 30, 30)
        >>> L = tm.uniform_laplacian()
        """

        # Check if connectivity information exists
        # If not, compute it
        if len(self.viv) == 0:
            self.compute_connectivity()

        # Get number of vertices in the mesh
        n = self.vertex.shape[0]

        # Initialize output matrix
        L = np.zeros((n, n))

        # Fill matrix entries
        for i in range(n):
            # Diagonal
            L[i, i] = -1.0
            # Get number of neighbors of this vertex
            num_neigh = float(len(self.viv[i]))
            # Add entry with uniform weight for each neighbor
            L[i, self.viv[i]] = 1.0/num_neigh

        return L

    # Create geometric Laplacian matrix
    def geometric_laplacian(self):
        """Create the geometric Laplacian operator for the mesh

        Parameters
        ----------
        None

        Returns
        -------
        L : numpy.array_like
            Laplacian matrix of dimensions n x n, where n is the number
            of vertices in the mesh
        negative : boolean
            Flag indicating whether any of the cotangent weights are
            negative
        boundary : boolean
            Flag indicating whether any boundaries where encountered in
            the mesh

        Notes
        -----
        The method constructors the discrete geometric Laplacian
        operator based on the geometry of the mesh.

        See Also
        --------
        geomproc.mesh.mesh.uniform_laplacian

        Examples
        --------
        >>> import geomproc
        >>> tm = geomproc.create_sphere(1.0, 30, 30)
        >>> L = tm.geometric_laplacian()
        """

        # Check if connectivity information exists
        # If not, compute it
        if (len(self.vif) == 0) or (len(self.viv) == 0):
            self.compute_connectivity()

        # Get number of vertices in the mesh
        n = self.vertex.shape[0]

        # Initialize output
        L = np.zeros((n, n))
        negative = 0
        boundary = 0

        # Fill matrix entries, with one row per vertex i
        for i in range(n):

            # Get vertex coordinates
            vi = self.vertex[i, :]

            # Initialize Voronoi area
            varea = 0

            # For each neighboring vertex j
            for j in self.viv[i]:
                # Get neighboring vertex j
                # (i, j) is the edge we are considering
                vj = self.vertex[j, :]

                # Get two other vertices that are not i or j
                #
                # Intersect faces of i and j to get the faces
                # that are common to i and j
                faces = list(set(self.vif[i]) & set(self.vif[j]))
                if len(faces) == 1:
                    # Mark boundary flag
                    boundary = 1
                    # Get all vertices of common face
                    vertices = set(self.face[faces[0], :])
                elif len(faces) >= 2:
                    # Get all vertices of common faces
                    vertices = set(self.face[faces[0], :]) | \
                               set(self.face[faces[1], :])
                # Keep only vertices other than i and j
                other = list(vertices - set([i, j]))

                # Now compute cotangents for mean curvature and the estimation
                # of Voronoi area
                cot = [0, 0]
                for k in range(len(other)):
                    # Get vertex coordinates
                    vk = self.vertex[other[k], :]
                    # Compute the cotangent of the corner k (vk)
                    vec1 = vi - vk
                    vec1 /= np.linalg.norm(vec1)
                    vec2 = vj - vk
                    vec2 /= np.linalg.norm(vec2)
                    cosine = np.dot(vec1, vec2)
                    sine = np.linalg.norm(np.cross(vec1, vec2))
                    cot[k] = cosine/sine
                    # Check if we have an obtuse angle
                    if cot[k] < 0:
                        # Use mixed area element
                        # Compute area of triangle
                        # cross / 2 to get area of triangle from area of parallelogram
                        # result / 2 to get 1/2 of the triangle area
                        cot[k] = np.linalg.norm(np.cross(vi - vk, vj - vk))/4

                # Define weight for this neighbor
                L[i, j] = cot[0] + cot[1]

                # Add Voronoi area estimate
                varea = varea + L[i, j]
                
                # Check negativity of cotan weights
                if L[i, j] < 0:
                    negative = 1

            # Weight all entries by area
            L[i, :] = (1/varea) * L[i, :]

            # Set diagonal
            L[i, i] = -L[i, :].sum()

        return [L, negative, boundary]

    # Remove duplicated vertices in the mesh
    def remove_duplicated_vertices(self, tol):
        """Remove duplicated vertices in the mesh

        Parameters
        ----------
        tol : float
            Tolerance for determining if two vertices are the same

        Returns
        -------
        None

        Notes
        -----
        The method removes duplicated vertices in the mesh by finding
        all vertices that have the same position up to the given
        tolerance, and keeping only one of the vertices in the mesh,
        updating vertex and face arrays and vertex references as
        necessary.
        """

        # Find closest points to each vertex and determine if they are close
        # enough according to the provided tolerance. If they are close
        # enough, they are marked to be merged together

        # Use a KDTree to find closest points efficiently
       
        # Create KDTree with all vertices
        index = [i for i in range(self.vertex.shape[0])]
        tree = KDTree(self.vertex.tolist(), index)

        # Create a map that tracks the merging of vertices
        # Initially, each vertex maps to itself
        # Vertices to be merged are then mapped to the vertex to be merged
        # with
        mp = [i for i in range(self.vertex.shape[0])]

        # Process each vertex
        for i in range(self.vertex.shape[0]):
            # If vertex has not been marked for merging yet
            if mp[i] == i:
                # Run KDTree query
                query = self.vertex[i]
                nearest = tree.nn_query(query, t=12) # Find nearest 12 points
                nm = 0
                # Process each nearest point
                for pnt in nearest:
                    # If nearest point is not itself
                    if pnt[1] != i:
                        # Compute distance to nearest point
                        dist = distance(pnt[0], query)
                        # If distance satisfies given tolerance
                        if dist < tol:
                            # Mark point for merging
                            mp[pnt[1]] = i
                            nm += 1

        # Now merge vertices together according to the map
        
        # Create new list of vertices
        new_vertex = []
        new_index = [0 for i in range(self.vertex.shape[0])]
        current_index = 0
        for i in range(len(mp)):
            # If vertex maps to itself, we keep it
            if mp[i] == i:
                new_vertex.append(self.vertex[i])
                new_index[i] = current_index
                current_index += 1

        # Create new list of faces
        new_face = []
        for i in range(len(self.face)):
            new_face.append([new_index[mp[self.face[i, 0]]], \
                             new_index[mp[self.face[i, 1]]], \
                             new_index[mp[self.face[i, 2]]]])

        # Update data arrays
        self.vertex = np.array(new_vertex)
        self.face = np.array(new_face, dtype=np.int_)

    # Add noise to the mesh
    def add_noise(self, scale):
        """Add noise to the vertex coordinates of a mesh

        Parameters
        ----------
        scale : float
            Scale that modulates the noise

        Returns
        -------
        None

        Notes
        -----
        For each vertex coordinate in the mesh, the method generates a
        random number between 0 and 1 and scales it by the 'scale'
        parameter. The scaled random number is then added to the
        coordinate value
        """

        self.vertex += scale*np.random.random(self.vertex.shape)
