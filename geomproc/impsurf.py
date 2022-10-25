#
# GeomProc: geometry processing library in python + numpy
#
# Copyright (c) 2008-2021 Oliver van Kaick <ovankaic@gmail.com>
# under the MIT License.
#
# See file LICENSE.txt for details on the copyright license.
#
"""This module contains the implicit function class of the GeomProc
geometry processing library used for defining implicit functions and
performing surface reconstruction.
"""


import numpy as np
import math
import random


# Implicit surface class
class impsurf:
    """A class that defines an implicit function

    Attributes
    ----------
    evaluate = pointer to a function(array_like x) : float
        Function used for evaluating the implicit function at a 3D point
        x, returning the signed distance of the surface to point x.

    Notes
    -----
    An implicit function can be setup by calling one of the setup_<name>
    methods. After that, the implicit function can be evaluated by
    simply calling the impsurf.evaluate(x) method.
    """

    def __init__(self):
        self.evaluate = None

    def compute_displaced_samples(self, pc, epsilon):
        """Create a set of samples displaced along point normals

        Parameters
        ----------
        pc : geomproc.pcloud
            Input point cloud stored as a point cloud object. Note that
            the point cloud needs to have normal vectors associated to
            the points
        epsilon : float
            Amount of displacement to perform along normals

        Returns
        -------
        None

        Notes
        -----
        Given an input point cloud, this method creates a set of samples
        that can be used for RBF surface reconstruction. Given an input
        point cloud with n points, the method creates a sample set with
        n*2 points, where n points are the original points from the
        input point cloud, and another n points are created by
        displacing each original sample along its normal by a value of
        epsilon. The samples are stored in the temporary attribute of
        the class called "sample", which is of shape (n*2, 3). Moreover,
        the method also creates a vector of displacements called
        "displacement", which is of shape (n*2, 1). The vector stores
        the displacement of each sample, which is zero for the original
        samples and epsilon for the new samples.

        See Also
        --------
        geomproc.impsurf.impsurf.setup_rbf
        """

        # Check if points have normals
        if pc.normal.shape[0] == 0:
            raise RuntimeError('point cloud does not have normals')

        # Get number of points in cloud
        n = pc.point.shape[0]

        # Initialize samples and their displacements
        self.sample = np.zeros((n*2, 3))
        self.displacement = np.zeros((n*2, 1))

        # The first portion of the samples are simply the points in the
        # point cloud with displacement 0
        self.sample[0:n, :] = pc.point

        # Add additional samples displaced from the surface by epsilon. The
        # samples are displaced along the normal direction
        for i in range(n):
            self.sample[n+i, :] = pc.point[i, :] + pc.normal[i, :]*epsilon
            self.displacement[n+i] = epsilon

    def compute_rbf(self, kernel, vectorized=False):
        """Reconstruct an implicit function from a set of point samples

        Parameters
        ----------
        kernel : function
            Kernel function of the form kernel(x, y) : float that
            computes the dissimilarity between two 3D points x and y,
            e.g., kernel = lambda x, y: math.pow(np.linalg.norm(x - y), 3)
        vectorized : boolean
            If vectorized is True, the method assumes that the kernel
            supplied function applies the kernel function to two sets of
            points, resulting in a matrix of shape (m, n) for sets of
            samples with m and n points. The default value of vectorized
            is False

        Returns
        -------
        None

        Notes
        -----
        The method reconstructs an implicit function from a set of point
        samples using the RBF method. The method assumes that a set of
        samples and displacements have been stored in the temporary
        attributes "sample" and "displacement", as described in the help
        of method surfrec.impsurf.compute_displaced_samples. The method
        then stores a temporary attribute "w" that represents the
        weights of radial basis functions (RBFs). The weights define the
        implicit function in the form phi(x) = \sum_{i=1}^n
        w(i)*kernel(x, sample(i)). The method also stores the given
        kernel in the temporary attribute "kernel".

        See Also
        --------
        geomproc.impsurf.impsurf.compute_displaced_samples
        geomproc.impsurf.impsurf.setup_rbf
        """

        # Check the type of kernel we are using
        if vectorized:
            # Apply vectorized kernel
            self.K = kernel(self.sample, self.sample)
            if self.K.shape != (self.sample.shape[0], self.sample.shape[0]):
                raise RuntimeError('vectorized kernel returns output of invalid size '+str(self.K.shape))
        else:
            # Get number of samples
            n = self.sample.shape[0]

            # Initialize matrix
            self.K = np.zeros((n, n))

            # Fill matrix entries
            for i in range(n):
                for j in range(n):
                    self.K[i, j] = kernel(self.sample[i, :], self.sample[j, :])


        # Solve linear system
        self.w = np.linalg.solve(self.K, self.displacement) 

        # Save kernel
        self.kernel = kernel

        # Remember kernel type
        self.vectorized = vectorized
       
    def evaluate_rbf(self, x):
        """Evaluate an implicit function encoded as an RBF

        Parameters
        ----------
        x : array_like
            3D point where the RBF should be evaluated

        Returns
        -------
        y : float
            Scalar value of the implicit function at point x

        Notes
        -----
        The method returns the value of the implicit function at a given
        point x. The value is typically the signed distance of the point
        to the surface. The method assumes that temporary attributes
        "sample", "kernel", and "w" have been stored in the class, as
        described in the help of methods
        surfrec.impsurf.compute_displaced_samples and surfrec.impsurf.compute_rbf.

        See Also
        --------
        geomproc.impsurf.impsurf.compute_displaced_samples
        geomproc.impsurf.impsurf.compute_rbf
        geomproc.impsurf.impsurf.setup_rbf
        """

        if self.vectorized:
            # Make sure input point is a row vector
            inx = np.array(x)
            if inx.shape[0] > 1:
                inx = x[np.newaxis, :]
            # Call kernel with all samples
            diff = self.kernel(inx, self.sample)
            # RBF
            y = np.sum(self.w*diff.T)
        else:
            y = 0.0
            for i in range(self.sample.shape[0]):
                y += self.w[i]*self.kernel(x, self.sample[i, :])
        
        return y

    def setup_rbf(self, pc, epsilon, kernel, vectorized=False):
        """Setup an implicit function based on a set of point samples

        Parameters
        ----------
        pc : geomproc.pcloud
            Input point cloud stored as a point cloud object. Note that
            the point cloud needs to have normal vectors associated to
            the points
        epsilon : float
            Amount of displacement to perform along normals
        kernel : function
            Kernel function of the form kernel(x, y) : float that
            computes the dissimilarity between two 3D points x and y,
            e.g., kernel = lambda x, y: math.pow(np.linalg.norm(x - y), 3)
        vectorized : boolean
            If vectorized is True, the method assumes that the kernel
            supplied function applies the kernel function to two sets of
            points, resulting in a matrix of shape (m, n) for sets of
            samples with m and n points. The default value of vectorized
            is False

        Returns
        -------
        None

        Notes
        -----
        Setup an implicit function by reconstructing the function from a
        set of point samples using the RBF method. The method first
        displaces the original point samples by a certain amount
        epsilon, to create additional samples that help avoid a trivial
        solution to the surface reconstruction problem. Then, the method
        reconstructs a surface with the RBF method based on the given
        kernel and solving a linear system. Once the implicit function
        is setup, it can be evaluated with the "evaluate" method of the
        class, which is a pointer to surfrec.impsurf.evalute_rbf.

        See Also
        --------
        geomproc.impsurf.impsurf.compute_displaced_samples
        geomproc.impsurf.impsurf.compute_rbf
        geomproc.impsurf.impsurf.evaluate_rbf
        """

        self.compute_displaced_samples(pc, epsilon)
        self.compute_rbf(kernel, vectorized)
        self.evaluate = self.evaluate_rbf

    def evaluate_sphere(self, p):
        """Evaluate the implicit function of a sphere

        Parameters
        ----------
        p : array_like
            3D point where the sphere should be evaluated

        Returns
        -------
        y : float
            Scalar value of the implicit function at point p

        Notes
        -----
        The method evaluates the implicit function of a sphere at a
        given point. The method assumes that the center and radius of
        the sphere have been stored in the temporary attributes "center"
        and "sphere" by the method surfrec.impsurf.setup_sphere.

        See Also
        --------
        geomproc.impsurf.impsurf.setup_sphere

        Examples
        --------
        >>> import geomproc
        >>> surf = geomproc.impsurf()
        >>> surf.setup_sphere(0.5)
        >>> val = surf.evaluate([0, 0, 0])
        """

        return ((p[0] - self.center[0])*(p[0] - self.center[0]) + 
                (p[1] - self.center[1])*(p[1] - self.center[1]) + 
                (p[2] - self.center[2])*(p[2] - self.center[2]) - 
                self.radius*self.radius)

    def setup_sphere(self, radius=1.0, center=[0.0, 0.0, 0.0]):
        """Setup the implicit function of a sphere

        Parameters
        ----------
        radius : float
            Scalar representing the radius of the sphere (the default
            value is 1)
        center : array_like
            3D point representing the center of the sphere (the default
            value is the origin)

        Returns
        -------
        None

        Notes
        -----
        The method sets up the implicit function for a sphere with a
        given center and radius. Once the implicit function is setup, it
        can be evaluated with the "evaluate" method of the class, which
        is a pointer to surfrec.evaluate_sphere.

        See Also
        --------
        geomproc.impsurf.impsurf.evaluate_sphere

        Examples
        --------
        >>> import geomproc
        >>> surf = geomproc.impsurf()
        >>> surf.setup_sphere(0.5)
        >>> val = surf.evaluate([0, 0, 0])
        """

        self.center = center
        self.radius = radius
        self.evaluate = self.evaluate_sphere

    def evaluate_torus(self, p):
        """Evaluate the implicit function of a torus

        Parameters
        ----------
        p : array_like
            3D point where the sphere should be evaluated

        Returns
        -------
        y : float
            Scalar value of the implicit function at point p

        Notes
        -----
        The method evaluates the implicit function of a torus at a given
        point. The method assumes that the two scalars "radius1" and
        "radius2" that describe the torus have been saved into temporary
        attributes of the class by the method
        surfrec.impsurf.setup_torus.

        See Also
        --------
        geomproc.impsurf.impsurf.setup_torus

        Examples
        --------
        >>> import geomproc
        >>> surf = geomproc.impsurf()
        >>> surf.setup_torus(0.6, 0.3)
        >>> val = surf.evaluate([0, 0, 0])
        """

        return math.pow(math.sqrt(p[0]*p[0] + p[1]*p[1]) - self.radius1, 2) + p[2]*p[2] - self.radius2*self.radius2

    def setup_torus(self, radius1, radius2):
        """Setup the implicit function of a torus

        Parameters
        ----------
        radius1 : float
            The distance from the center of the tube to the center of the torus
        radius2: float
            Radius of the tube

        Returns
        -------
        None

        Notes
        -----
        The method sets up the implicit function for a torus which is
        radially symmetric about the z-axis. Once the implicit function
        is setup, it can be evaluated with the "evaluate" method of the
        class, which is a pointer to surfrec.evaluate_torus.

        See Also
        --------
        geomproc.impsurf.impsurf.evaluate_torus

        Examples
        --------
        >>> import geomproc
        >>> surf = geomproc.impsurf()
        >>> surf.setup_torus(0.6, 0.3)
        >>> val = surf.evaluate([0, 0, 0])
        """

        self.radius1 = radius1
        self.radius2 = radius2
        self.evaluate = self.evaluate_torus
