import geomproc

import utilities

import math

import time

import os

# Import numpy library for handling matrices
import numpy as np

class algorithm_parameters:
    def __init__(self, v_x, v_y, v_z, min_plane_size, min_points_per_voxel, angle_threshold, neighbors_per_point): 
        self.voxel_x_size = v_x
        self.voxel_y_size = v_y
        self.voxel_z_size = v_z
        self.min_plane_size = min_plane_size
        self.min_points_per_voxel = min_points_per_voxel
        self.angle_threshold = angle_threshold
        self.neighbors_per_point = neighbors_per_point

def determine_max_x_y_z_values(pc):
    min_x = math.inf
    max_x = -math.inf
    min_y = math.inf
    max_y = -math.inf
    min_z = math.inf
    max_z = -math.inf
    for i in range(pc.point.shape[0]):
        if pc.point[i][0] < min_x:
            min_x = pc.point[i][0]
        elif pc.point[i][0] > max_x:
            max_x = pc.point[i][0]
        elif pc.point[i][1] < min_y:
            min_y = pc.point[i][1]
        elif pc.point[i][1] > max_y:
            max_y = pc.point[i][1]
        elif pc.point[i][2] < min_z:
            min_z = pc.point[i][2]
        elif pc.point[i][2] > max_z:
            max_z = pc.point[i][2]
    return min_x, max_x, min_y, max_y, min_z, max_z


def x_y_partition_points(pc, points_to_partition, min_x, max_x, min_y, max_y, alg_params):
    num_voxels_along_x = int((max_x - min_x)/alg_params.voxel_x_size) + 1
    num_voxels_along_y = int((max_y - min_y)/alg_params.voxel_y_size) + 1
    x_y_partition = [[[] for j in range(num_voxels_along_y)] for i in range(num_voxels_along_x)]
    for point_to_partition in points_to_partition:
        voxel_to_update_x = math.floor((pc.point[point_to_partition][0]-min_x)/alg_params.voxel_x_size)
        voxel_to_update_y = math.floor((pc.point[point_to_partition][1]-min_y)/alg_params.voxel_y_size)
        x_y_partition[voxel_to_update_x][voxel_to_update_y].append(point_to_partition)
    return x_y_partition
        
def get_neighbouring_x_y_voxels(x_y_partition, current_voxel, num_voxels_along_x, num_voxels_along_y):
    neighbouring_x_y_voxels = []
    if current_voxel[0] + 1 < num_voxels_along_x:
        neighbouring_x_y_voxels.append([current_voxel[0] + 1, current_voxel[1]])
    if current_voxel[0] - 1 >= 0:
        neighbouring_x_y_voxels.append([current_voxel[0] - 1, current_voxel[1]])
    if current_voxel[1] + 1 < num_voxels_along_y:
        neighbouring_x_y_voxels.append([current_voxel[0], current_voxel[1] + 1])
    if current_voxel[1] - 1 >= 0:
        neighbouring_x_y_voxels.append([current_voxel[0], current_voxel[1] - 1])
    return neighbouring_x_y_voxels

def find_seed_x_y_voxel(is_voxel_visited, x_y_partition, min_points_per_voxel, num_visited_voxels, prev_seed):
    i = prev_seed[0]
    for j in range(prev_seed[1], is_voxel_visited.shape[1]):
        if is_voxel_visited[i,j] == 0:
            is_voxel_visited[i,j] = 1
            num_visited_voxels = num_visited_voxels + 1
            if len(x_y_partition[i][j]) >= min_points_per_voxel:
                return i,j, num_visited_voxels
    
    for i in range(prev_seed[0] + 1, is_voxel_visited.shape[0]):
        for j in range(0, is_voxel_visited.shape[1]):
            if is_voxel_visited[i,j] == 0:
                is_voxel_visited[i,j] = 1
                num_visited_voxels = num_visited_voxels + 1
                if len(x_y_partition[i][j]) >= min_points_per_voxel:
                    return i,j, num_visited_voxels
                    
    return -1,-1, math.inf

def validate_neighbouring_x_y_voxels(neighbouring_x_y_voxels, x_y_partition, next_voxels_to_visit, is_in_next_voxels_to_visit, min_points_per_voxel, is_voxel_visited, pc, min_z_component, filter_using_normals):
    for neighbour in neighbouring_x_y_voxels:
        if is_voxel_visited[neighbour[0],neighbour[1]] == 0 and is_in_next_voxels_to_visit[neighbour[0],neighbour[1]] == 0:
            if len(x_y_partition[neighbour[0]][neighbour[1]]) >= min_points_per_voxel:
                if filter_using_normals:
                    for point in x_y_partition[neighbour[0]][neighbour[1]]:
                        num_horizontal_points = 0
                        if filter_point_based_on_normal(pc, point, min_z_component):
                            num_horizontal_points = num_horizontal_points + 1
                        if num_horizontal_points >= min_points_per_voxel:
                            next_voxels_to_visit.append(neighbour)
                            is_in_next_voxels_to_visit[neighbour[0],neighbour[1]] = 1
                            break
                else:
                    next_voxels_to_visit.append(neighbour)
                    is_in_next_voxels_to_visit[neighbour[0],neighbour[1]] = 1
                    
        
def find_planes_at_layer(x_y_partition, alg_params, pc, min_z_component, filter_using_normals):
    num_voxels_along_x = len(x_y_partition)
    num_voxels_along_y = len(x_y_partition[0])
    min_plane_size_in_voxels = alg_params.min_plane_size / (alg_params.voxel_x_size * alg_params.voxel_y_size)
    if min_plane_size_in_voxels > num_voxels_along_x * num_voxels_along_y:
        print("plane size is too big to find")
        return False, 0
    num_visited_voxels = 0
    is_voxel_visited = np.zeros((num_voxels_along_x,num_voxels_along_y))
    
    planes = []
    prev_seed = [0,0]
    while True:
        plane = np.zeros((num_voxels_along_x,num_voxels_along_y))
        plane_size = 0
        is_in_next_voxels_to_visit = np.zeros((num_voxels_along_x,num_voxels_along_y))
        next_voxels_to_visit = []
        i, j, num_visited_voxels = find_seed_x_y_voxel(is_voxel_visited, x_y_partition, alg_params.min_points_per_voxel, num_visited_voxels, prev_seed)
        current_voxel = [i,j]
        prev_seed = [i,j]
        if num_visited_voxels >= num_voxels_along_x * num_voxels_along_y:
            break
        
        while True:
            plane[current_voxel[0],current_voxel[1]] = plane[current_voxel[0],current_voxel[1]] + 1
            plane_size = plane_size + 1
            neighbouring_x_y_voxels = get_neighbouring_x_y_voxels(x_y_partition, current_voxel, num_voxels_along_x, num_voxels_along_y)
            validate_neighbouring_x_y_voxels(neighbouring_x_y_voxels, x_y_partition, next_voxels_to_visit, is_in_next_voxels_to_visit, alg_params.min_points_per_voxel, is_voxel_visited, pc, min_z_component, filter_using_normals)
            if len(next_voxels_to_visit) == 0:
                break
            num_visited_voxels = num_visited_voxels + 1
            current_voxel = next_voxels_to_visit.pop(0)
            is_in_next_voxels_to_visit[current_voxel[0],current_voxel[1]] = 0
            is_voxel_visited[current_voxel[0],current_voxel[1]] = 1
        
        if plane_size > min_plane_size_in_voxels:
            print("found a plane with " + str(plane_size) + " voxels")
            planes.append(plane)
            
    if len(planes) == 0:
        return False, 0
    else:
        return True, planes

def calculate_minimum_z_component(angle_threshold):
    return np.cos(angle_threshold)

def filter_point_based_on_normal(pc, point, min_z_component):    
    if abs(pc.normal[point][2]) > min_z_component:
        return True
    return False
        

"""Detects support surfaces in a point cloud using a voxel-based region growing
approach. Each layer of the voxel grid is processed one at a time in 
sequential order. This method can run much faster if implemented with parallel
processing in mind, i.e. process multiple levels in parallel.

Parameters
----------
pc : pcloud
    input point cloud
file_name : string
    string used for creating the names of files to save the output point clouds
    to
alg_params : algorithm_parameters
    the algorithm parameters corresponding to the ones described in the 
    corresponding written report
sample_size : int
    sample fraction of points to sample for running in the region based growing 
    algorithm
filter_using_normals : bool
    whether or not points should be filtered based on their estimated normals
    should be used

Returns
-------
Time that the algorithm took to complete. It also outputs 2 point clouds, one 
called 'X not detected.obj' and one called 'X detected.obj', where X is the 
name of the input file to the program.

Notes
-----
The method estimates the normal of each point in the point cloud
by performing an analysis of covariance on the k nearest
neighbors of each point. The method sets the 'normal' attribute
of the class with the estimated normals.

"""
def detect_planes(pc, file_name, alg_params, sample_size = 1.0, filter_using_normals = True):
    pc.point = utilities.sample_point_cloud(pc, round(sample_size * pc.point.shape[0]))
    if filter_using_normals:
        start = time.process_time()
        pc.estimate_normals_simple(alg_params.neighbors_per_point)
        time_to_complete = time.process_time() - start
        print("total time to estimate normals: " + str(time_to_complete) + " seconds")
    min_z_component = calculate_minimum_z_component(alg_params.angle_threshold)
    
    start = time.process_time()
    
    min_x, max_x, min_y, max_y, min_z, max_z = determine_max_x_y_z_values(pc)
    range_of_x = max_x - min_x
    range_of_y = max_y - min_y
    range_of_z = max_z - min_z
    print("number of points: " + str(pc.point.shape[0]))
    print("range of x values: " + str(range_of_x))
    print("range of y values: " + str(range_of_y))
    print("range of z values: " + str(range_of_z))
    num_voxels_along_x = range_of_x/alg_params.voxel_x_size
    num_voxels_along_y = range_of_y/alg_params.voxel_y_size
    num_voxels_along_z = range_of_z/alg_params.voxel_z_size
    print("number of voxels along x: " + str(num_voxels_along_x))
    print("number of voxels along y: " + str(num_voxels_along_y))
    print("number of voxels along z: " + str(num_voxels_along_z))
    print("total number of voxels: " + str(num_voxels_along_x * num_voxels_along_y  * num_voxels_along_z))
    print("voxels at a given height: " + str(num_voxels_along_x * num_voxels_along_y))
    print("voxels needed to be in a plane: " + str(alg_params.min_plane_size / (alg_params.voxel_x_size * alg_params.voxel_y_size)))
    
    num_z_voxels = int((max_z - min_z)/alg_params.voxel_z_size) + 1
    #the maximum height point is in a voxel by itself in the case that v_z evenly divides max_z - min_z
    z_voxels = [[] for i in range(num_z_voxels)]
    for i in range(pc.point.shape[0]):
        index_to_update = math.floor((pc.point[i][2]-min_z)/alg_params.voxel_z_size) 
        z_voxels[index_to_update].append(i)

    not_on_plane_mask = np.ones(pc.point.shape[0])
    points_on_planes = []
    #process each layer of the voxel grid one at a time
    for i in range(len(z_voxels)):
        x_y_partition = x_y_partition_points(pc, z_voxels[i], min_x, max_x, min_y, max_y, alg_params)
        succeeded, planes = find_planes_at_layer(x_y_partition, alg_params, pc, min_z_component, filter_using_normals)
        if succeeded:
            for plane in planes:
                for i in range(len(x_y_partition)):
                    for j in range(len(x_y_partition[0])):
                        if plane[i,j] == 1:
                            for point in x_y_partition[i][j]:
                                points_on_planes.append(point)
                                not_on_plane_mask[point] = 0
    
    pc_of_planes = geomproc.pcloud()
    points_on_planes_coordinates = []
    
    for point in points_on_planes:
        points_on_planes_coordinates.append(pc.point[point].tolist())
    
    time_to_complete = time.process_time() - start
    print("total time to detect planes: " + str(time_to_complete) + " seconds")
    
    pc_of_planes.point = np.array(points_on_planes_coordinates)

    wo = geomproc.write_options()
    wo.write_vertex_colors = True
    wo.write_face_normals = False
    file_name, extension = os.path.splitext(file_name)
    
    pc_of_points_not_on_planes = geomproc.create_points(pc.point[not_on_plane_mask.astype(bool)], radius=0, color = [0,1,0])
    pc_of_points_not_on_planes.save(file_name + ' not detected' + extension, wo)
    pc_of_points_on_planes = geomproc.create_points(pc_of_planes.point, radius=0, color = [1,0,0])
    pc_of_points_on_planes.save(file_name + ' detected' + extension, wo)
    return time_to_complete
    

#amount to sample the point cloud by
sample_size = 1
#number of neighbors used per point when estimating normals
neighbors_per_point = 15
#angle threshold for determining if a normal is vertically facing or not
angle_threshold = 0.5


file_to_detect_planes_in = 'table_1.obj'
mesh = geomproc.load(file_to_detect_planes_in)
pc = mesh.sample(20000)
utilities.swap_y_z_coordinates(pc)
#the size of each voxel along the x, y and z dimensions respectively.
voxel_x_size = 2
voxel_y_size = 2
voxel_z_size = 2
#the minimum area of plane for detecting support surfaces
min_plane_size = 10000
#the minimum number of points that must be in a voxel for it to be counted as a part of a support surface
min_points_per_voxel = 1
#choose whether or not normal estimation and filtering using the estimated normals will occur
filter_using_normals = False

file_to_detect_planes_in = '2 chairs and a table 2.obj'
pc =  geomproc.load(file_to_detect_planes_in)
pc.add_noise(0.05)
voxel_z_size = 0.05
voxel_x_size = 0.1
voxel_y_size = 0.1
min_plane_size = 0.5
min_points_per_voxel = 1
angle_threshold = 0.6
neighbors_per_point = 30
sample_size = 1.0
filter_using_normals = False

file_to_detect_planes_in = 'simple_shelf_scan_improved.obj'
pc =  geomproc.load(file_to_detect_planes_in)
pc.add_noise(0.05)
voxel_z_size = 0.07
voxel_x_size = 0.1
voxel_y_size = 0.1
min_plane_size = 2.0
min_points_per_voxel = 1
detected_p_radius = 0
angle_threshold = 0.5
neighbors_per_point = 30
sample_size = 1.0
filter_using_normals = False

alg_params = algorithm_parameters(voxel_x_size, voxel_y_size, voxel_z_size, min_plane_size, min_points_per_voxel, angle_threshold, neighbors_per_point)
detect_planes(pc, file_to_detect_planes_in, alg_params,sample_size, filter_using_normals)

# num_points_list = list(range(5000,500000,5000))
# time_to_compute_list = []
# for num_points in num_points_list:
#     file_to_detect_planes_in = 'table_1.obj'
#     mesh = geomproc.load(file_to_detect_planes_in)
#     pc = mesh.sample(num_points)
#     utilities.swap_y_z_coordinates(pc)
#     voxel_z_size = 1
#     voxel_x_size = 10
#     voxel_y_size = 10
#     min_plane_size = 10000
#     min_points_per_voxel = 1
#     detected_p_radius = 1
#     filter_using_normals= False

#     alg_params = algorithm_parameters(voxel_x_size, voxel_y_size, voxel_z_size, min_plane_size, min_points_per_voxel, angle_threshold, neighbors_per_point)
#     time_to_compute_list.append(detect_planes(pc, file_to_detect_planes_in, alg_params,sample_size, filter_using_normals))
    

# # print(num_points_list)
# # print(time_to_compute_list)
# import matplotlib.pyplot as plt



# plt.plot(num_points_list, time_to_compute_list)
# plt.title('Support Surface Detection Processing Time vs. Number of Points')
# plt.xlabel('Number of Points in the Point Cloud')
# plt.ylabel('Time to Run Support Surface Detection (seconds)')
# plt.savefig('C:/School/Geometry Course/Final Project/Code/examples to show in written report/time vs points.png', format='png', bbox_inches='tight',pad_inches = 0, dpi=900)
# plt.show()



# num_voxels_list = list(range(5000,5000,5000))
# time_to_compute_list = []
# for num_voxels in num_voxels_list:
#     file_to_detect_planes_in = 'table_1.obj'
#     mesh = geomproc.load(file_to_detect_planes_in)
#     pc = mesh.sample(100000)
#     utilities.swap_y_z_coordinates(pc)
#     min_x, max_x, min_y, max_y, min_z, max_z = determine_max_x_y_z_values(pc)
#     num_voxels_along_each_axis = np.power(num_voxels, (1/3))
#     voxel_x_size = (max_x - min_x)/num_voxels_along_each_axis
#     voxel_y_size = (max_y - min_y)/num_voxels_along_each_axis
#     voxel_z_size = (max_z - min_z)/num_voxels_along_each_axis
#     min_plane_size = 10000
#     min_points_per_voxel = 1
#     detected_p_radius = 1
#     filter_using_normals= False

#     alg_params = algorithm_parameters(voxel_x_size, voxel_y_size, voxel_z_size, min_plane_size, min_points_per_voxel, angle_threshold, neighbors_per_point)
#     time_to_compute_list.append(detect_planes(pc, file_to_detect_planes_in, alg_params,sample_size, filter_using_normals))
    

# # print(num_points_list)
# # print(time_to_compute_list)
# import matplotlib.pyplot as plt



# plt.plot(num_voxels_list, time_to_compute_list)
# plt.title('Support Surface Detection Processing Time vs. Number of Voxels')
# plt.xlabel('Number of Voxels in the Point Cloud')
# plt.ylabel('Time to Run Support Surface Detection (seconds)')
# #plt.savefig('C:/School/Geometry Course/Final Project/Code/examples to show in written report/time vs num_voxels.png', format='png', bbox_inches='tight',pad_inches = 0, dpi=900)
# plt.show()