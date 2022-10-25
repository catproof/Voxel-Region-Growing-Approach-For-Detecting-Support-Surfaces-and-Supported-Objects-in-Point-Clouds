#
# KDTree
# Copyright Matej Drame [matej.drame@gmail.com]
# https://code.google.com/archive/p/python-kdtree/downloads
#
# Modified by Oliver van Kaick (2021) to include the following features:
# - Include an index (backlink) for each point
# - Perform range and distance queries
#
#
# GeomProc: geometry processing library in python + numpy
#
# Copyright (c) 2008-2021 Oliver van Kaick <ovankaic@gmail.com>
# under the MIT License.
#
# See file LICENSE.txt for details on copyright licenses.
#
"""This module contains a KDTree implementation for use in the GeomProc
geometry processing library.
"""

def square_distance(pointA, pointB):
    # squared euclidean distance
    distance = 0
    dimensions = len(pointA) # assumes both points have the same dimensions
    for dimension in range(dimensions):
        distance += (pointA[dimension] - pointB[dimension])**2
    return distance

class KDTreeNode():
    def __init__(self, point, index, left, right):
        self.point = point
        self.index = index
        self.left = left
        self.right = right
    
    def is_leaf(self):
        return (self.left == None and self.right == None)

class KDTreeNeighbours():
    # Internal structure used in nearest-neighbours search.

    def __init__(self, query_point, t, had_index=False):
        self.query_point = query_point
        self.t = t # neighbours wanted
        self.largest_distance = 0 # squared
        self.current_best = []
        self.had_index = had_index

    def calculate_largest(self):
        if self.t >= len(self.current_best):
            self.largest_distance = self.current_best[-1][2]
        else:
            self.largest_distance = self.current_best[self.t-1][2]

    def add(self, point, index):
        sd = square_distance(point, self.query_point)
        # run through current_best, try to find appropriate place
        for i, e in enumerate(self.current_best):
            if i == self.t:
                return # enough neighbours, this one is farther, let's forget it
            if e[2] > sd:
                self.current_best.insert(i, [point, index, sd])
                self.calculate_largest()
                return
        # append it to the end otherwise
        self.current_best.append([point, index, sd])
        self.calculate_largest()
    
    def get_best(self):
        if self.had_index:
            return [element[0:2] for element in self.current_best[:self.t]]
        else:
            return [element[0] for element in self.current_best[:self.t]]
        
class KDTree():
    """A class that implements a KDTree

    Notes
    -----
    The class implements a KDTree that can be used for different types
    of points queries: nearest neighbor search, range search, and
    distance search. The dimension of the queries should be the same as
    the dimension of the points that were used to build the tree. 

    Examples
    --------
    from kdtree import KDTree
    
    data = <load data> # iterable of points (which are also iterable, same length)
    index = index of each data point to a reference array
    point = <the point of which neighbours we're looking for>
    
    tree = KDTree(data, index)
    nearest = tree.query(point, t=4) # find nearest 4 points
    """
    
    def __init__(self, data, index=[]):
        self.data = data
        self.index = index
        """KDTree constructor

        Parameters
        ----------
        data : list
            A list of length n with the input points that are used for
            building the KDTree. Each point is also represented as a
            list. It is assumed that all the points have the same number
            of dimensions
        index : list, optional
            A list of length n with the index or backlink of each point
            in the 'data' list
        """

        def build_kdtree(point_list, index, depth):
            # code based on wikipedia article: http://en.wikipedia.org/wiki/Kd-tree
            if not point_list:
                return None

            # select axis based on depth so that axis cycles through all valid values
            axis = depth % len(point_list[0]) # assumes all points have the same dimension

            # sort point list and choose median as pivot point,
            # TODO: better selection method, linear-time selection, distribution
            alldata = list(zip(point_list, index))
            #point_list.sort(key=lambda point: point[axis])
            alldata.sort(key=lambda point: point[0][axis])
            [point_list, index] = zip(*alldata)
            median = int(len(point_list)/2) # choose median

            # create node and recursively construct subtrees
            node = KDTreeNode(point=point_list[median], 
                              index=index[median],
                              left=build_kdtree(point_list[0:median], index[0:median], depth+1),
                              right=build_kdtree(point_list[median+1:], index[median+1:], depth+1))
            return node
        
        if not index:
            index = [i for i in range(len(data))]
            self.had_index = False
        else:
            self.had_index = True

        self.root_node = build_kdtree(data, index, depth=0)

        # Calculate min and max for each dimension to define the region
        # covered by the KDTree
        dim = len(data[0])
        min_val = [float('inf') for i in range(dim)]
        max_val = [float('-inf') for i in range(dim)]
        for i in range(len(data)):
            for j in range(dim):
                if data[i][j] < min_val[j]:
                    min_val[j] = data[i][j]
                if data[i][j] > max_val[j]:
                    max_val[j] = data[i][j]
        self.region = [min_val, max_val]
    
    def nn_query(self, query_point, t=1):
        """Perform a nearest-neighbors query with the KDTree

        Parameters
        ----------
        query_point : list
            Query point with the same number of dimensions as the points
            in the KDTree
        t : int
            Number of nearest neighbors to retrieve

        Returns
        -------
        result : list
            List of nearest neighbor points. If the tree was constructed
            with an index associated to each point, then the list
            contains tuples of the form (point, index). Otherwise, the
            result is simply a list of points
        """

        statistics = {'nodes_visited': 0, 'far_search': 0, 'leafs_reached': 0}
        
        def nn_search(node, query_point, t, depth, best_neighbours):
            if node == None:
                return
            
            #statistics['nodes_visited'] += 1
            
            # if we have reached a leaf, let's add to current best neighbours,
            # (if it's better than the worst one or if there is not enough neighbours)
            if node.is_leaf():
                #statistics['leafs_reached'] += 1
                best_neighbours.add(node.point, node.index)
                return
            
            # this node is no leaf
            
            # select dimension for comparison (based on current depth)
            axis = depth % len(query_point)
            
            # figure out which subtree to search
            near_subtree = None # near subtree
            far_subtree = None # far subtree (perhaps we'll have to traverse it as well)
            
            # compare query_point and point of current node in selected dimension
            # and figure out which subtree is farther than the other
            if query_point[axis] < node.point[axis]:
                near_subtree = node.left
                far_subtree = node.right
            else:
                near_subtree = node.right
                far_subtree = node.left

            # recursively search through the tree until a leaf is found
            nn_search(near_subtree, query_point, t, depth+1, best_neighbours)

            # while unwinding the recursion, check if the current node
            # is closer to query point than the current best,
            # also, until t points have been found, search radius is infinity
            best_neighbours.add(node.point, node.index)
            
            # check whether there could be any points on the other side of the
            # splitting plane that are closer to the query point than the current best
            # Note from Oliver: changed < to <=, to find identical points
            if (node.point[axis] - query_point[axis])**2 <= best_neighbours.largest_distance:
                #statistics['far_search'] += 1
                nn_search(far_subtree, query_point, t, depth+1, best_neighbours)
            
            return
        
        # if there's no tree, there's no neighbors
        if self.root_node != None:
            neighbours = KDTreeNeighbours(query_point, t, self.had_index)
            nn_search(self.root_node, query_point, t, depth=0, best_neighbours=neighbours)
            result = neighbours.get_best()
        else:
            result = []
        
        #print statistics
        return result

    def range_query(self, query):
        """Perform a range query with the KDTree

        Parameters
        ----------
        query : list
            Range for performing a query with the KDTree, which is an
            n-dimensional box given as a list [minimum, maximum], where
            minimum and maximum are n-dimensional points

        Returns
        -------
        result : list
            List of all the points inside the given range. If the tree
            was constructed with an index associated to each point, then
            the list contains tuples of the form (point, index).
            Otherwise, the result is simply a list of points
        """

        # Check if point is contained in the query box
        def point_contained(point, query):
            for j in range(len(point)):
                if (point[j] < query[0][j]) or (point[j] > query[1][j]):
                    return False
            return True

        # Check if the given region is fully contained in the query box
        def region_fully_contained(region, query):
            for j in range(len(query[0])):
                if (region[0][j] < query[0][j]) or (region[1][j] > query[1][j]):
                    return False
            return True

        # Check if the given region intersects the query box
        def region_intersects(region, query):
            for j in range(len(query[0])):
                if (region[0][j] > query[1][j]) or (region[1][j] < query[0][j]):
                    return False
            return True

        # Report entire subtree (no need to perform any tests to select
        # the output nodes)
        def report_subtree(node, had_index, result):
            # Check if the subtree does not exist
            if node == None:
                return

            # Add the current node to the result
            if had_index:
                result.append((node.point, node.index))
            else:
                result.append(node.point)

            # Stop recursion at a leaf node
            if node.is_leaf():
                return

            # Continue including nodes on the left and right subtrees
            report_subtree(node.left, had_index, result)
            report_subtree(node.right, had_index, result)

        def range_search(node, region, query, dim, depth, had_index, result):
            # Check if the subtree does not exist
            if node == None:
                return

            # Check if current point should be reported
            if point_contained(node.point, query):
                if had_index:
                    result.append((node.point, node.index))
                else:
                    result.append(node.point)

            # Stop recursion at a leaf node
            if node.is_leaf():
                return

            # Select dimension for comparison (based on current depth)
            axis = depth % dim

            # Create deep copies of the current region
            left_region = [[val for val in region[0]], [val for val in region[1]]]
            right_region = [[val for val in region[0]], [val for val in region[1]]]

            # Check whether we should continue recursion along the left
            # subtree
            left_region[1][axis] = node.point[axis]
            if region_fully_contained(left_region, query):
                report_subtree(node.left, had_index, result)
            elif region_intersects(left_region, query):
                range_search(node.left, left_region, query, dim, depth+1, had_index, result)

            # Check whether we should continue recursion along the right
            # subtree
            right_region[0][axis] = node.point[axis]
            if region_fully_contained(right_region, query):
                report_subtree(node.right, had_index, result)
            elif region_intersects(right_region, query):
                range_search(node.right, right_region, query, dim, depth+1, had_index, result)

        # Initialize output
        result = []
        
        # Perform search if we have a root node
        if self.root_node != None:
            range_search(self.root_node, self.region, query, len(self.root_node.point), 0, self.had_index, result)

        return result

    def dist_query(self, center, radius):
        """Perform a distance query with the KDTree

        Parameters
        ----------
        center : list
        radius : float
            Information for performing the distance query, where center
            is an n-dimensional point and radius is a scalar

        Returns
        -------
        result : list
            List of all the points that are located a distance of radius
            or less from the center point. If the tree was constructed
            with an index associated to each point, then the list
            contains tuples of the form (point, index). Otherwise, the
            result is simply a list of points
        """

        # Create cube so that the sphere is inscribed in the cube
        cube = [[i for i in center], [i for i in center]] # Deep copy
        for j in range(len(center)):
            cube[0][j] = center[j] - radius
            cube[1][j] = center[j] + radius

        # Perform range query with the cube
        temp = self.range_query(cube)

        # Filter out points that do not satisfy the distance
        result = []
        radius_sq = radius**2
        for i in range(len(temp)):
            if self.had_index:
                if square_distance(center, temp[i][0]) <= radius_sq:
                    result.append(temp[i])
            else:
                if square_distance(center, temp[i]) <= radius_sq:
                    result.append(temp[i])

        return result
