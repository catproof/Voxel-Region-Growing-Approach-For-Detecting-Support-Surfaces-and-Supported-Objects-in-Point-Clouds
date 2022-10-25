#
# GeomProc: geometry processing library in python + numpy
#
# Copyright (c) 2008-2021 Oliver van Kaick <ovankaic@gmail.com>
# under the MIT License.
#
# See file LICENSE.txt for details on copyright licenses.
#
"""This module contains graph functions of the GeomProc geometry
processing library.
"""


import heapq


# A data structure for storing shortest paths
class shortest_path:
    """A class that stores shortest path information

    Attributes
    ----------
    sources : list
        List of all source nodes that were used to compute shortest path
        information, where sources[i] is the i-th source
    dist : matrix
        Matrix of path distances for each source, where dist[i] is a
        list of distances from the i-th source to all other nodes in the
        graph, with dist[i][j] being the distance from sources[i] to
        node j in the graph
    pred : matrix
        Matrix of shortest path predecessors for each source, where
        pred[i][j] is the predecessor of node j in the shortest paths
        starting at source[i]

    Notes
    -----
    An object of this class is created when computing shortest paths in
    the graph class. The distances of the paths can be retrieved from
    the 'dist' attribute, while the paths can be obtained with the
    get_path method.
    """

    def __init__(self):
        self.sources = []
        self.dist = []
        self.pred = []

    def get_path(self, source_index, target):
        """Get shortest path between a source and target node

        Parameters
        ----------
        source_index : int
            Index of the source in the 'sources' attributes, indicating
            which source we are using to obtain the path
        target : int
            Target node for the path

        Returns
        -------
        path : list
            List with the sequence of nodes that provides the path from
            sources[i] to 'target'

        Notes
        -----
        The method returns a shortest path from node
        sources[source_index] to node 'target'
        """

        node = target
        path = []
        while node != -1:
            path.insert(node)
            node = pred[source_index][node]
        return path


# A graph data structure
class graph:
    """A class that represents a sparse directed graph

    Attributes
    ----------
    adj : list of lists
        List of node adjacencies, where adj[i] is a list of all the
        nodes that are connected to node 'i'. 
    weight : list of lists
        List of edge weights, where weight[i] is a list of all the
        weights of the edges to the neighbors of node 'i', according to
        the order of neighbors in 'adj'. For example, weight[i][j] is
        the weight of edge (i, adj[i][j])

    Notes
    -----
    The graph stores a directed graph. Thus, if node j appears in
    adj[i], it does not imply that node i appears in adj[j]. If the goal
    is to construct an undirected graph, the symmetric relation has to
    be manually enforced by updating both lists.
    """

    def __init__(self):
        # Initialize all attributes
        # Adjacency list
        self.adj = []
        # Weight list
        self.weight = []

    # Deep copy
    def copy(self):
        """Perform a deep copy of the graph

        Parameters
        ----------
        None

        Returns
        -------
        g : graph
            New copied graph
        """

        g = graph()
        g.adj = self.adj.copy()
        g.weight = self.weight.copy()
        return g

    def num_nodes():
        """Number of nodes in the graph

        Parameters
        ----------
        None

        Returns
        -------
        number_of_nodes : int
            Number of nodes in the graph
        """

        return len(self.adj)

    def append(self):
        """Append a node to the graph

        Parameters
        ----------
        None

        Returns
        -------
        Index of node appended to the graph
        """

        self.adj.append([])
        self.weight.append([])
        return len(self.adj)-1

    def is_neighbor(self, i, j):
        """Check if two nodes are neighbors in the graph

        Parameters
        ----------
        i : int
            Index of first node
        j : int
            Index of second node

        Returns
        -------
        relation : boolean
            Whether the nodes are neighbors. Note that the result of
            is_neighbor(i, j) may be different from is_neighbor(j, i),
            since the graph is directed
        """

        return j in self.adj[i]

    def get_weight(self, i, j, default=0):
        """Get the weight between two neighbors in the graph

        Parameters
        ----------
        i : int
            Index of first node
        j : int
            Index of second node
        default : variant
            Weight to be returned by default if the nodes are not
            connected

        Returns
        -------
        weight : variant
            Weight of the edge between the two neighbors. Note that the
            result of get_weight(i, j) may be different from
            get_weight(j, i), since the graph is directed
        """

        if j in self.adj[i]:
            index = self.adj[i].index(j)
            return self.weight[i][index]
        else:
            return default

    def get_neighbors(self, i):
        """Get a list of all neighbors of a node

        Parameters
        ----------
        i : int
            Index of the node

        Returns
        -------
        neighbors : list
            List of all neighbors of node i
        """

        return self.adj[i]

    def get_weights(self, i):
        """Get a list of the weights to all the neighbors of a node

        Parameters
        ----------
        i : int
            Index of the node

        Returns
        -------
        weights : list
            List of weights of the edges from 'i' to all its neighbors,
            following the order provided by the 'adj' attribute or the
            output of get_neighbors
        """

        return self.weight[i]

    def set_neighbor(self, i, j, weight=1):
        """Add a neighbor to a node

        Parameters
        ----------
        i : int
            Index of first node
        j : int
            Index of neighbor
        weight : variant
            Weight of the edge (i, j). The default value is 1

        Returns
        -------
        None
        """
 
        if i != j:
            if not j in self.adj[i]:
                self.adj[i].append(j)
                self.weight[i].append(weight)

    def set_neighbors(self, i, neigh, weight=1):
        """Add multiple neighbors to a node

        Parameters
        ----------
        i : int
            Index of node
        neigh : list
            List with the indices of nodes to be added as neighbors
        weight : variant
            Weight of the edges from node 'i' to the new neighbors. The
            parameter can be either a single scalar that is assigned as
            the weight of all the edges, or a list of the same length as
            'neigh' with the individual edge weights. The default weight
            is 1

        Returns
        -------
        None
        """

        if (len(weight) != 1) and (len(weight) != len(neigh)):
            raise RuntimeError('weight parameter should be of length 1 or the same length as the neigh vector')
        for index in range(len(neigh)):
            j = neigh[index]
            if j != i:
                if j not in self.adj[i]:
                    self.adj[i].append(j)
                    if len(weight) == 1:
                        self.weight[i].append(weight)
                    else:
                        self.weight[i].append(weight[index])

    def grow_minimum_spanning_tree(self, root, visited):
        # Grow a minimum spanning tree from the provided root
        # Variable 'visited' is modified in place

        # Initialize output tree
        tree = []
        # Initialize heap with an edge from an undefined node (-1) to
        # the root
        h = []
        heapq.heappush(h, (0.0, -1, root))
        # Process edges while the heap is not empty
        while len(h) > 0:
            # Pop edge from the heap
            element = heapq.heappop(h)
            parent = element[1]
            current = element[2]
            # Check if node has not been visited yet
            if not visited[current]:
                # Mark node as visited
                visited[current] = True
                # Add edge to the spanning tree
                if parent != -1:
                    tree.append((parent, current))
                # Add neighbors of node to the heap
                for index in range(len(self.adj[current])):
                    j = self.adj[current][index]
                    weight = self.weight[current][index]
                    heapq.heappush(h, (weight, current, j))
        # Return spanning tree
        return tree

    def compute_minimum_spanning_forest(self, root=0):
        """Compute a minimum spanning forest for the graph

        Parameters
        ----------
        root : int
            Index of the node to be used as the root of the first
            spanning tree

        Returns
        -------
        forest : list of lists
            Forest of all the minimum spanning trees computed on the
            graph. Forest[i] is the i-th spanning tree, which is a list
            of tuples denoting the edges in the tree

        Notes
        -----
        The method starts growing a spanning tree from each node in the
        graph that has not been reached yet by a tree. This is necessary
        in case we have multiple connected components in the graph. The
        set of all spanning trees forms the spanning forest. The first
        tree is grown from the suggested root provided as a parameter.
        """

        # Initialize list marking visited nodes
        visited = [False for item in range(len(self.adj))]
        # Grow a tree from the provided root
        tree = self.grow_minimum_spanning_tree(root, visited)
        # Initialize output forest with the first tree
        forest = [tree]
        # Grow trees for remaining nodes that have not been visited
        for index in range(len(visited)):
            # Check if not visited yet
            if not visited[index]:
                # Grow spanning tree from this node
                tree = self.grow_minimum_spanning_tree(index, visited)
                forest.append(tree)
        # Return spanning forest
        return forest

    def compute_shortest_paths(self, sources):
        """Compute shortest paths for a set of source nodes

        Parameters
        ----------
        sources : list of int
            Indices of the nodes to be used as sources for computing
            shortest paths

        Returns
        -------
        sp : geomproc.shortest_path object
            Object that stores the shortest path information and can be
            used to retrieve paths or path lengths

        See Also
        --------
        geomproc.graph.shortest_path

        Notes
        -----
        The method computes shortest paths on the graph with Dijkstra's
        algorithm, implemented with a priority queue.
        """

        # Check type of input for 'sources'
        # If input is a single number, transform it into a list
        if not type(sources) is list:
            sources = [sources]
        # Initialize output object
        sp = shortest_path()
        # Initialize sources
        sp.sources = sources
        # Initialize table of distances for all sources
        inf = float('inf')
        sp.dist = [[inf for i in range(len(self.adj))] for j in range(len(sources))]
        # Initialize table of predecessors
        sp.pred = [[-1 for i in range(len(self.adj))] for j in range(len(sources))]
        # Compute shortest paths from each source
        for source_index, source in enumerate(sources):
            # Initialize heap with source element
            h = []
            heapq.heappush(h, (0.0, source))
            # Set distance of source to zero
            sp.dist[source_index][source] = 0.0
            # Process heap while not empty
            while len(h) > 0:
                # Pop element with minimum distance from the heap
                current = heapq.heappop(h)[1]
                # Look at neighbors of popped element
                for index in range(len(self.adj[current])):
                    # Get neighbor index and weight
                    neigh = self.adj[current][index]
                    weight = self.weight[current][index]
                    # Compute distance when going through the popped
                    # element (alternative distance)
                    alt_dist = sp.dist[source_index][current] + weight
                    # Compare the alternative distance to the distance
                    # in the table
                    if (sp.dist[source_index][neigh] > alt_dist):
                        # Update distance if the alternative is shorter
                        sp.dist[source_index][neigh] = alt_dist
                        # Add neighbor to the queue
                        heapq.heappush(h, (alt_dist, neigh))
                        # Update path
                        sp.pred[source_index][neigh] = current

        return sp
