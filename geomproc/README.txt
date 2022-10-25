Updated GeomProc code can be found here: https://github.com/ovankaic/GeomProc        

	GeomProc: geometry processing library in python + numpy

                              Version 1.0
                             April 30, 2021

     Copyright (c) 2008-2021 Oliver van Kaick <ovankaic@gmail.com>
        http://people.scs.carleton.ca/~olivervankaick/index.html
                    Released under the MIT License.
    For all the parts of the code where authorship is not explicitly
                               mentioned.

 Marching cubes code based on C++ code written by Matthew Fisher (2014)
      (https://graphics.stanford.edu/~mdfisher/MarchingCubes.html)
         License: unknown. Highly similar C++ code exists with
       Copyright (C) 2002 by Computer Graphics Group, RWTH Aachen
                         under the GPL license.

                            KDTree in python
             Copyright Matej Drame [matej.drame@gmail.com]
       https://code.google.com/archive/p/python-kdtree/downloads

        See file LICENSE.txt for details on copyright licenses.

GeomProc is a geometry processing library intended for educational
purposes. Thus, the library was developed with an emphasis on legibility
of the code, documentation, and ease of use, rather than efficiency,
although the efficiency should not lag behind similar implementations in
interpreted languages, as the included methods are based on
state-of-the-art algorithms. However, there is no guarantee that the
code is as efficient as a C++ implementation, or applicable to large
triangle meshes. To ensure ease of use, the library has only one
dependency: python with numpy. Note that an external mesh viewer such as
MeshLab is required for visualizing the output of the library.

The library comprises a set of example implementations of geometry
processing methods applicable to triangle meshes, point clouds, and
implicit functions. The basic classes of the library implement triangle
mesh and point cloud data structures. The library was developed from
code initially written in Matlab.

Please refer to the documentation in the "doc" directory and example
scripts in the current directory.
