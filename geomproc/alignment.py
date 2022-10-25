#
# GeomProc: geometry processing library in python + numpy
#
# Copyright (c) 2008-2021 Oliver van Kaick <ovankaic@gmail.com>
# under the MIT License.
#
# See file LICENSE.txt for details on copyright licenses.
#
"""This module contains functions for aligning point clouds with the
GeomProc geometry processing library.
"""


import numpy as np
from .pcloud import *
from .kdtree import *


def closest_points(pc1, pc2, err_type=1):
    """Find closest points from one point cloud to another

    Parameters
    ----------
    pc1 : pcloud
        First set of points

    pc2 : pcloud
        Second set of points

    err_type : int, optional
        Type of error to be calculated by the function. The error type
        can be one of: 0 = maximum error; 1 = average error.
        The default value is 1

    Returns
    -------
    corr : array_like
        Correspondence of points from pc1 to their closest points in
        pc2, represented as an array of shape (n, 3), where 'n' is the
        number of points in pc1, and each correspondence is of the form
        [index of point in pc1, index of corresponding point in pc2,
        distance between points]

    Notes
    -----
    For each point in pc1, the function finds the closest point in pc2,
    and sets the output correspondence with this match. The function
    also calculates the error of the correspondences, according to the
    specified error type
    """

    # Initialize correspondence and error
    corr = np.zeros((pc1.point.shape[0], 3), dtype=np.int_)
    err = 0

    # Set up KDTree with all points
    index = [i for i in range(pc2.point.shape[0])]
    tree = KDTree(pc2.point.tolist(), index)

    # Find closest points and compute error
    for i in range(pc1.point.shape[0]):
        # Find point in pc2 that is closest to point 'i' in pc1
        nearest = tree.nn_query(pc1.point[i, :], 1)
        min_dist = distance(pc1.point[i, :], nearest[0][0])
        corr[i, :] = [i, nearest[0][1], min_dist]

        # Quadratic code
        if 0:
            min_dist = float('inf') 
            for j in range(pc2.point.shape[0]):
                # Compute Euclidean distance between points
                dist = np.linalg.norm(pc1.point[i, :] - pc2.point[j, :])
                # Compare to current minimum distance
                if dist < min_dist:
                    # We found a closer point, update min_dist and
                    # correspondence
                    min_dist = dist
                    corr[i, :] = [i, j, min_dist]

        # Add error for point to total error
        if err_type == 0:
            # Max error
            err = max(err, min_dist)
        elif err_type == 1:
            # Average error
            err += min_dist

    # Get average
    if err_type == 1:
        err /= pc1.point.shape[0]

    return [corr, err]


def filter_correspondences(corr, keep):
    """Filter a set of correspondences according to the match distances

    Parameters
    ----------
    corr : array_like
        Correspondence of points from one point cloud to another,
        represented as an array of shape (n, 3), where 'n' is the number
        of matches, and each match is of the form [index of point in
        first point cloud, index of corresponding point in second point
        cloud, distance between points]

    keep : float
        Percentage of closest points to be kept

    Returns
    -------
    corr : array_like
        Filtered correspondence in the same format as the input 'corr'

    Notes
    -----
    The function filters the input correspondences by keeping only the
    top 'keep' matches with the smallest distances
    """

    # Keep only best quality correspondences
    corr_new = corr.copy()
    if keep < 1.0:
        corr_new = list(corr_new)
        corr_new.sort(key=lambda match: match[2])
        corr_new = corr_new[0:int(math.ceil(len(corr_new)*keep))]
        corr_new = np.array(corr_new, dtype=np.int_)
    return corr_new


def transformation_from_correspondences(pc1, pc2, corr):
    """Estimate a transformation from a set of corresponding points

    Parameters
    ----------
    pc1 : pcloud
        First set of points

    pc2 : pcloud
        Second set of points

    corr : array_like
        Correspondence of points from pc1 to pc2, represented as an
        array of shape (n, 3), where 'n' is the number of
        correspondences, and each correspondence is of the form [index
        of point in pc1, index of corresponding point in pc2, distance
        between points]. The distance between points is not used by this
        function and can be set to 0 if needed

    Returns
    -------
    rot : array_like
        3D rotation matrix, represented as an array of shape (3, 3)

    trans : array_like
        3D translation vector, represented as a vector of shape (3, 1)

    Notes
    -----
    The function estimates the rigid transformation that best aligns two
    sets of points in a least-squares sense based on a set of point
    correspondences.
    The implementation is based on the report of Sorkine-Hornung and
    Rabinovich, "Least-Squares Rigid Motion Using SVD", ETH Zurich
    Technical Report, 2017.
    """

    # Create two corresponding sets
    x1 = pc1.point[corr[:, 0], :]
    x2 = pc2.point[corr[:, 1], :]

    # Compute centroids
    c1 = np.mean(x1, axis=0, keepdims=True)
    c2 = np.mean(x2, axis=0, keepdims=True)

    # Compute centered vectors
    x1 = x1 - c1
    x2 = x2 - c2

    # Compute covariance matrix
    covar = np.dot(x1.T, x2)

    # Compute SVD of variance
    [u, s, v] = np.linalg.svd(covar)
    # Note that svd in numpy returns the matrix v already transposed

    # Derive transformation: rotation and translation
    dm = np.diag([1, 1, np.linalg.det(np.dot(v, u))])
    rot = np.dot(np.dot(v.T, dm), u.T)
    trans = c2.T - np.dot(rot, c1.T)

    return [rot, trans]


def apply_transformation(x, rot, trans):
    """Apply a rigid transformation to a set of 3D points

    Parameters
    ----------
    x : array_like
        Set of 3D points, represented as an array of shape (n, 3), where
        'n' is the number of points in the set

    rot : array_like
        3D rotation matrix, represented as an array of shape (3, 3)

    trans : array_like
        3D translation vector, represented as a vector of shape (3, 1)

    Returns
    -------
    y : array_like
        Set of transformed points, represented as an array of shape 
        (n, 3)
    """

    # Apply (rot, trans) to each point of x
    y = x.copy()
    for i in range(x.shape[0]):
        y[i, :] = (np.dot(rot, x[i, :, np.newaxis]) + trans).T
    return y


# Similar to apply_transformation, but directly modifies the input x
def apply_transformation_in_place(x, rot, trans):
    """Apply a rigid transformation to a set of 3D points in place

    Parameters
    ----------
    x : array_like
        Set of 3D points, represented as an array of shape (n, 3), where
        'n' is the number of points in the set

    rot : array_like
        3D rotation matrix, represented as an array of shape (3, 3)

    trans : array_like
        3D translation vector, represented as a vector of shape (3, 1)

    Returns
    -------
    None

    Notes
    -------
    This function is similar to :py:func:`geomproc.alignment.apply_transformation`,
    but modifies the input x with the transformation
    """

    # Apply (rot, trans) to each point of x
    for i in range(x.shape[0]):
        x[i, :] = (np.dot(rot, x[i, :, np.newaxis]) + trans).T


# Complete ICP method for aligning two point clouds
def icp(pc1, pc2, error_threshold, max_iter, keep=1.0):
    """Align two point clouds with the Iterative Closest Points (ICP) method

    Parameters
    ----------
    pc1 : pcloud
        One of the two point clouds to be aligned

    pc2 : pcloud
        The other point cloud to be aligned

    error_threshold : float
        Error to be satisfied by the alignment

    max_iter : int
        Maximum number of iterations to be run, which takes precedence
        over the error_threshold parameter. Can be set to float('inf')

    keep : float, optional
        Percentage of closest points to be used for estimating the best
        transformation that aligns one point cloud to the other. The
        default is to keep all correspondences (keep = 1.0)

    Returns
    -------
    rot : array_like
        3D rotation matrix for the computed transformation, which is an
        array of shape (3, 3)

    trans : array_like
        3D translation vector for the computed transformation, which is
        a vector of shape (3, 1)

    pc1tr : pcloud
        The input point cloud pc1 transformed by (rot, trans) so that it
        aligns with the input pc2

    err : float
        The error that was satisfied at the end of the algorithm

    iter_count : int
        Number of iterations that were carried out by the algorithm

    corr : array_like
        Final correspondence of points from pc1 to pc2, represented as
        an array of shape (n, 3), where 'n' is the number of
        correspondences, and each correspondence is of the form [index
        of point in pc1, index of corresponding point in pc2, distance
        between points]

    Notes
    -----
    The ICP algorithm aligns two point clouds pc1 and pc2 with an
    iterative algorithm. Each iteration consists in finding the closest
    point in pc2 to each point in pc1, estimating a rotation and
    translation that best aligns pc1 to pc2 in a least-squares sense
    according to the correspondences, and then applying the
    transformation to pc1. The algorithms ends when either the input
    error threshold is satisfied or the maximum number of iterations is
    reached. The error computed by the function represents how close one
    point cloud is to the other after alignment
    """

    # Initialize rotation, translation, and iteration count
    rot = np.identity(3)
    trans = np.zeros((3, 1))
    iter_count = 0

    # Initialize output point cloud
    pc1tr = pc1.copy()

    #### Standard ICP loop
    # For each point in pc1, find its closest point pc2
    # Also compute the error for the current alignment
    [corr, err] = closest_points(pc1tr, pc2)
    # If need to iterate
    while err > error_threshold:
        # Keep only best quality correspondences
        if keep < 1.0:
            corr = filter_correspondences(corr, keep)
        # Derive a rigid transformation from the point correspondences
        [rot, trans] = transformation_from_correspondences(pc1tr, pc2, corr)
        # Apply the transformation to the point cloud in place
        apply_transformation_in_place(pc1tr.point, rot, trans)
        # Recompute closest points and error
        [corr, err] = closest_points(pc1tr, pc2)
        # Check number of iterations used
        iter_count = iter_count + 1
        if iter_count > max_iter:
            break

    return [rot, trans, pc1tr, err, iter_count, corr]


# Options for computing spin images
class spin_image_options:
    """A class that holds the configuration for computing spin image descriptors

    Attributes
    ----------
    radius : float
        The size of the spin image along its radial direction

    height : float
        The size of the spin image along its height. In analogy to the
        radius, the height is effectively the length from the center of
        the spin image to one of its extremities, so that height*2 is
        the length of the entire spin image

    radial_bins : int
        Number of bins along the radial direction

    height_bins : int
        Number of bins along the height direction

    normalize : boolean
        Normalize each bin by the total number of votes in all the bins,
        resulting in a descriptor with floating-point values in the
        bins. This results in a descriptor that is more comparable
        across models with different numbers of samples

    Notes
    -----
    The resulting descriptor is an array of size radial_bins*height_bins
    """

    def __init__(self):
        # The selected parameters are different from the original spin
        # image paper: the parameters are set to create more localized
        # descriptors
        self.radius = 0.5
        self.height = 0.5
        self.radial_bins = 10
        self.height_bins = 10
        self.normalize = True


def spin_images(pc1, pc2, opt = spin_image_options()):
    """Compute spin image descriptors for a point cloud

    Parameters
    ----------
    pc1 : pcloud
        The function computes a spin image descriptor for each point in
        the point cloud pc1

    pc2 : pcloud
        The points in the point cloud pc2 are used for casting votes
        when constructing the spin image descriptors. pc2 can simply be
        the same as pc1. However, typically it will be a larger set of
        points than pc1, so that the descriptors can be computed with
        enough detail. In any case, pc1 and pc2 should be sampled from
        the same shape

    opt : geomproc.spin_image_options, optional
        Object with the configuration for computing the spin image
        descriptors

    Returns
    -------
    desc : array_like
        Spin image descriptors, represented as an array of shape (n,
        radial_bins*height_bins), where 'n' is the number of points in
        pc1, and radial_bins*height_bins is the total number of bins in
        one descriptor according to the given configuration object.
        desc[i, :] represents the descriptor of point 'i' in pc1

    See Also
    --------
    geomproc.spin_image_options

    Notes
    -----
    The implementation is based on the paper of Johnson and Hebert,
    "Using Spin Images for Efficient Object Recognition in Cluttered 3D
    Scenes", IEEE PAMI 21(5), 1999.
    To compute one spin image descriptor, the method places a cylinder
    at a point according to the position of the point and orientation of
    the normal of the point. It then divides the cylinder radially and
    along its normal to create a number of bins, and counts how many
    points fall inside each bin. Finally, if desired, each bin is
    normalized by the total number of points in all the bins, to make
    the descriptor more robust to point clouds with different numbers of
    samples.
    """

    # Initialize descriptor
    desc = np.zeros((pc1.point.shape[0], opt.radial_bins*opt.height_bins))

    # Set up KDTree with all points from pc2
    tree = KDTree(pc2.point.tolist())

    # Build descriptor for each point in pc1
    for i in range(pc1.point.shape[0]):
        # Get point and its normal
        point = pc1.point[i, :]
        normal = pc1.normal[i, :]
        # Get all the points in the range of the descriptor (neighbors)
        neighbors = tree.dist_query(pc1.point[i, :], opt.radius)
        # Iterate through each neighbor
        for j in range(len(neighbors)):
            # Get neighbor
            neigh = np.array(neighbors[j])
            #### Compute radial and height distances for this neighbor
            # Form a vector from the reference point to the neighbor
            vec = neigh - point
            # Project the vector on the normal of the reference point
            # to get the distance of the neighbor along the normal
            # Also, normalize the distance by the height of the
            # descriptor
            height_dist = np.dot(normal, vec) / opt.height
            # Project the vector on the plane perpendicular to the
            # normal to get the distance of the neighbor along the
            # radial direction
            # Also, normalize the distance by the radius of the
            # descriptor
            radial_dist = np.linalg.norm(vec - height_dist*normal) / opt.radius
            # Check if point is inside the range of the descriptor and
            # can be considered in the descriptor construction
            # Since we normalized the distances by radius and height, we
            # can simply compare to 1.0
            if (radial_dist < 1.0) and (abs(height_dist) < 1.0):
                # Normalize the height_dist to a value between 0 and 1
                height_dist = (height_dist + 1.0)/2.0
                # Find bin index for radial and height distances
                radial_index = math.floor(radial_dist*opt.radial_bins)
                height_index = math.floor(height_dist*opt.height_bins)
                # Convert two bin indices into one index and cast a vote
                # in the corresponding bin
                desc[i, radial_index + height_index*opt.radial_bins] += 1

    # If normalizing, divide each bin by the total number of votes in
    # all the bins
    if opt.normalize:
        desc /= desc.sum()

    return desc


def best_match(desc1, desc2):
    """Find the best match for two sets of descriptors

    Parameters
    ----------
    desc1 : array_like
        First set of descriptors, given as an array of shape (n, d),
        where 'n' is the number of descriptors in the set, and 'd' the
        number of dimensions in one descriptor

    desc2 : array_like
        Second set of descriptors, in the same format as desc1

    Returns
    -------
    corr : array_like
        Matching of descriptors, represented as an array of shape (n,
        3), where 'n' is the number of descriptors in desc1, and each
        correspondence is of the form [index of descriptor in desc1,
        index of corresponding descriptor in desc2, distance between
        descriptors]

    Notes
    -----
    For each descriptor in desc1, the function finds the best matching
    descriptor in desc2 (with the smallest descriptor distance) by
    computing the Euclidean distance between descriptors
    """

    # Initialize output correspondence
    corr = np.zeros((desc1.shape[0], 3), dtype=np.int_)

    # Find best descriptor match
    for i in range(desc1.shape[0]):
        min_dist = float('inf')
        for j in range(desc2.shape[0]):
            # Compute Euclidean distance
            dist = np.linalg.norm(desc1[i, :] - desc2[j, :])
            # Check if this is the best distance so far
            if dist < min_dist:
                min_dist = dist
                corr[i, :] = [i, j, min_dist]

    # Return correspondence
    return corr
