#
# GeomProc: geometry processing library in python + numpy
#
# Copyright (c) 2008-2021 Oliver van Kaick <ovankaic@gmail.com>
# under the MIT License.
#
# See file LICENSE.txt for details on copyright licenses.
#
"""This module contains all the classes and functions of the GeomProc
geometry processing library.
"""


import numpy as np

from .write_options import *
from .mesh import *
from .graph import *
from .kdtree import *


# A point cloud data structure
class pcloud:
    """A class that represents a point cloud

    Attributes
    ----------
    point : numpy.array_like
        Points in the point cloud. The array should be of shape (n, 3),
        where n is the number of points in the point cloud. Each row of
        the array stores one point, and the columns of the array
        represent the x, y, and z coordinates of a point.

    normal : numpy.array_like
        Point normals. The array should be either empty (to indicate
        that this attribute is not present) or of shape (n, 3), where n
        is the number of points in the point cloud. The i-th row of the
        array stores the normal vector for the i-th point in the point
        cloud, and the columns of the array are the x, y, and z
        components of the normal vector.

    color : numpy.array_like
        Point colors. The array should be either empty (to indicate that
        this attribute is not present) or of shape (n, 3), where n is
        the number of points in the point cloud. The i-th row of the
        array stores the RGB color for the i-th point in the point
        cloud in the order r, g, and b.
    """

    def __init__(self):
        # Initialize all attributes
        # Points
        self.point = np.zeros((0, 3), dtype=np.single)
        # Normals
        self.normal = np.zeros((0, 3), dtype=np.single)
        # Colors
        self.color = np.zeros((0, 3), dtype=np.single)

    # Deep copy
    def copy(self):
        """Perform a deep copy of the point cloud

        Parameters
        ----------
        None

        Returns
        -------
        pc : pcloud
            New copied point cloud
        """

        pc = pcloud()
        pc.point = self.point.copy()
        pc.normal = self.normal.copy()
        pc.color = self.color.copy()
        return pc

    # Save a point cloud to a file
    def save(self, filename, wo = write_options()): 
        """Save a point cloud to a file

        Parameters
        ----------
        filename : string
            Name of the output filename
        wo : write_options object, optional
            Object with flags that indicate which fields of the point
            cloud should be written to the output file

        Returns
        -------
        None

        Notes
        -----
        The method saves the point cloud information into a file. The
        file format is determined from the filename extension.
        Currently, the obj and ply file formats are supported. By
        default, only points are written into the file. Other
        information is written if the corresponding flags are set in the
        write_options object. Not all flags are supported by all file
        formats.

        See Also
        --------
        geomproc.write_options

        Examples
        --------
        >>> import geomproc
        >>> tm = geomproc.create_sphere(1.0, 30, 30)
        >>> pc = tm.sample(1000)
        >>> pc.save('sphere_samples.obj')
        """

        # Check the file extension and call the relevant method to save
        # the file
        part = filename.split('.')
        if part[-1].lower() == 'obj':
            return self.save_obj(filename, wo)
        elif part[-1].lower() == 'ply':
            return self.save_ply(filename, wo)
        else:
            raise RuntimeError('file format "'+part[-1]+'" not supported')

    # Save a point cloud to a file in obj format
    def save_obj(self, filename, wo = write_options()):

        # Open the file
        with open(filename, 'w') as f: 
            # Write points
            if wo.write_point_colors:
                for i in range(self.point.shape[0]):
                    f.write('v '+str(self.point[i, 0])+' '+
                            str(self.point[i, 1])+' '+
                            str(self.point[i, 2])+' '+
                            str(self.color[i, 0])+' '+
                            str(self.color[i, 1])+' '+
                            str(self.color[i, 2])+'\n')
            else:
                for i in range(self.point.shape[0]):
                    f.write('v '+str(self.point[i, 0])+' '+
                            str(self.point[i, 1])+' '+
                            str(self.point[i, 2])+'\n')

            # Write list of normals
            if wo.write_point_normals:
                for i in range(self.normal.shape[0]):
                    f.write('vn '+str(self.normal[i, 0])+' '+
                            str(self.normal[i, 1])+' '+
                            str(self.normal[i, 2])+'\n')

    # Save a point cloud to a file in off format
    def save_ply(self, filename, wo = write_options()):
        # Open the file
        with open(filename, 'w') as f: 
            # Write header
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex '+str(self.point.shape[0])+'\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            if wo.write_point_normals:
                f.write('property float nx\n')
                f.write('property float ny\n')
                f.write('property float nz\n')
            if wo.write_point_colors:
                f.write('property float r\n')
                f.write('property float g\n')
                f.write('property float b\n')
            f.write('end_header\n')
            # Write data
            for i in range(self.point.shape[0]):
                f.write(str(self.point[i, 0])+' '+
                        str(self.point[i, 1])+' '+
                        str(self.point[i, 2]))
                if wo.write_point_normals:
                    f.write(' '+str(self.normal[i, 0])+' '+
                                str(self.normal[i, 1])+' '+
                                str(self.normal[i, 2]))
                if wo.write_point_colors:
                    f.write(' '+str(self.color[i, 0])+' '+
                                str(self.color[i, 1])+' '+
                                str(self.color[i, 2]))
                f.write('\n')

    def estimate_normals_simple(self, k):
        """Estimate point normals from nearest neighbors

        Parameters
        ----------
        k : int
            Number of nearest neighbors to use in the estimate

        Returns
        -------
        None

        Notes
        -----
        The method estimates the normal of each point in the point cloud
        by performing an analysis of covariance on the k nearest
        neighbors of each point. The method sets the 'normal' attribute
        of the class with the estimated normals.

        Examples
        --------
        >>> import geomproc
        >>> tm = geomproc.create_sphere(1.0, 30, 30)
        >>> pc = tm.sample(1000)
        >>> pc.estimate_normals_simple(10)
        """

        # Intialize normal vectors
        self.normal = np.zeros((self.point.shape[0], 3))

        # Use a kdtree to find nearest neighbors
        tree = KDTree(self.point.tolist())

        # Iterate through each point
        for i in range(self.point.shape[0]):
            # Get nearest neighbors (including the point itself)
            nearest = tree.nn_query(self.point[i, :], k+1)
            nearest = np.array(nearest)

            # Compute covariance matrix of neighbors
            cov = np.cov(nearest, rowvar=False)

            # Compute eigenvalues and eigenvectors
            [eigenval, eigenvec] = np.linalg.eig(cov)

            # Sort eigenvectors, since numpy does not do that
            idx = eigenval.argsort()[::-1]   
            eigenval = eigenval[idx]
            eigenvec = eigenvec[:,idx]

            # Get normal as eigenvector associated to smallest
            # eigenvalue
            self.normal[i, :] = eigenvec[:, 2]
            
        # Add noise to the mesh
    def add_noise(self, scale):
        """Add noise to the point coordinates of a point cloud

        Parameters
        ----------
        scale : float
            Scale that modulates the noise

        Returns
        -------
        None

        Notes
        -----
        For each point coordinate in the point cloud, the method generates a
        random number between 0 and 1 and scales it by the 'scale'
        parameter. The scaled random number is then added to the
        coordinate value
        """
        #print(scale*np.random.random(self.point.shape))
        self.point += scale*np.random.random(self.point.shape)

    def estimate_normals(self, k, consistent=-1):
        """Estimate point normals from nearest neighbors

        Parameters
        ----------
        k : int
            Number of nearest neighbors to use in the estimate
        consistent : int, optional
            Index of a point that we know has a consistent normal. If
            this parameter is not provided, it is estimated with a
            heuristic

        Returns
        -------
        None

        Notes
        -----
        The method estimates the normal of each point in the point cloud
        by performing an analysis of covariance on the k nearest
        neighbors of each point. Then, the method flips the directions
        of the normals consistently. The method sets the 'normal'
        attribute of the class with the estimated normals.

        Examples
        --------
        >>> import geomproc
        >>> tm = geomproc.create_sphere(1.0, 30, 30)
        >>> pc = tm.sample(1000)
        >>> pc.estimate_normals(10)
        """

        # Check if normal2 should be flipped to be more consistent with
        # the orientation of normal1
        def check_flip(normal1, normal2):
            dp = np.dot(normal1, normal2)
            if dp < 0.0:
                return True
            return False

        # Create knn graph
        # The graph is used both to find nearest neighbors of vertices
        # for covariance computation and to flip normals consistently
        g = graph()
        g.adj = [[] for i in range(self.point.shape[0])]
        g.weight = [[] for i in range(self.point.shape[0])]
        # Use a kdtree to find nearest neighbors
        index = [i for i in range(self.point.shape[0])]
        tree = KDTree(self.point.tolist(), index)
        for i in range(self.point.shape[0]):
            # Get nearest neighbors (including the point itself)
            nearest = tree.nn_query(self.point[i, :], k+1)
            # Process each nearest point
            for pnt in nearest:
                # Exclude the point itself
                if pnt[1] != i:
                    g.adj[i].append(pnt[1])
                    g.weight[i].append(distance(self.point[i, :], pnt[0]))

        # Intialize normal vectors
        self.normal = np.zeros((self.point.shape[0], 3))

        # Iterate through each point
        for i in range(self.point.shape[0]):
            # Get nearest neighbors and the point itself
            nearest = g.adj[i].copy() # Avoid modifying adj
            nearest.append(i)
            nearest = self.point[nearest, :]

            # Compute covariance matrix of neighbors
            cov = np.cov(nearest, rowvar=False)

            # Compute eigenvalues and eigenvectors
            [eigenval, eigenvec] = np.linalg.eig(cov)

            # Sort eigenvectors, since numpy does not do that
            idx = eigenval.argsort()[::-1]   
            eigenval = eigenval[idx]
            eigenvec = eigenvec[:,idx]

            # Get normal as eigenvector associated to smallest
            # eigenvalue
            self.normal[i, :] = eigenvec[:, 2]

        # Consistently flip normals

        # Make graph symmetric to improve connectedness
        for i in range(self.point.shape[0]):
            for j in g.adj[i]:
                if not i in g.adj[j]:
                    g.adj[j].append(i)
                    g.weight[j].append(distance(self.point[i, :], 
                                                self.point[j, :]))

        if consistent == -1:
            # Find corner point of the point cloud
            corner = self.point.min(axis=0)
            center = np.average(self.point, axis=0)

            # Create vector pointing from center of point cloud to corner
            vector = corner - center
            vector /= np.linalg.norm(vector)
            
            # Find point in the cloud closest to the corner
            min_dist = float('inf')
            min_index = -1
            for i in range(self.point.shape[0]):
                dist = distance(self.point[i, :], corner)
                if dist < min_dist:
                    min_dist = dist
                    min_index = i
            consistent = min_index
            # Flip normal of point closest to corner if necessary
            if check_flip(self.normal[consistent, :], vector):
                self.normal[consistent, :] *= -1.0

        # Compute minimum spanning forest of knn graph starting from the
        # corner which may have a more consistent normal
        forest = g.compute_minimum_spanning_forest(consistent)

        # Flip normals according to each tree
        for tree in forest:
            for edge in tree:
                # Flip normal, if necessary
                if check_flip(self.normal[edge[0], :], self.normal[edge[1], :]):
                    self.normal[edge[1], :] *= -1.0
