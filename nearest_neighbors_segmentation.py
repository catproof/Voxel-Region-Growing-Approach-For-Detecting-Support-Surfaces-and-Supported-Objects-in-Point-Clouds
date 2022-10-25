import geomproc

import utilities

import math

import time

import os

# Import numpy library for handling matrices
import numpy as np

    
#need to make this more efficient
def find_seed_point(is_point_visited, num_visited_points, prev_seed):
    for i in range(prev_seed, is_point_visited.shape[0]):
        if is_point_visited[i] == 0:
            is_point_visited[i] = 1
            num_visited_points = num_visited_points + 1
            return i, num_visited_points
                    
    return -1, math.inf

def get_neighbouring_points(pc, current_point, tree, num_neighbors):
    nearest = tree.nn_query(pc.point[current_point], num_neighbors)
    nearest_point_indices = []
    for i in nearest:
        nearest_point_indices.append(i[1])
    return nearest_point_indices

def validate_neighbouring_points(neighbouring_points, next_points_to_visit, is_in_next_points_to_visit, is_point_visited):
    for neighbour in neighbouring_points:
        if is_point_visited[neighbour] == 0 and is_in_next_points_to_visit[neighbour] == 0:
            next_points_to_visit.append(neighbour)
            is_in_next_points_to_visit[neighbour] = 1

def segment_points(pc, file_name, num_neighbors, sample_size = 1.0, max_num_shapes_to_save = 10):
    pc.point = utilities.sample_point_cloud(pc, round(sample_size * pc.point.shape[0]))
    print("make a k-d tree with " + str(pc.point.shape[0]) + " points")
    
    start_time = time.process_time()
    
    tree = geomproc.KDTree(pc.point.tolist(),list(range(pc.point.shape[0])))
    print("done making the k-d tree")
    
    shapes = []

    num_visited_points = 0
    is_point_visited = np.zeros(pc.point.shape[0])
    
    prev_seed = 0
    while True:
        shape = []
        is_in_next_points_to_visit = np.zeros(pc.point.shape[0])
        next_points_to_visit = []
        i, num_visited_points = find_seed_point(is_point_visited, num_visited_points, prev_seed)
        current_point = i
        prev_seed = i
        if num_visited_points >= pc.point.shape[0]:
            break
        
        while True:
            shape.append(current_point)
            neighbouring_points = get_neighbouring_points(pc, current_point, tree, num_neighbors)
            validate_neighbouring_points(neighbouring_points, next_points_to_visit, is_in_next_points_to_visit, is_point_visited)
            if len(next_points_to_visit) == 0:
                break
            num_visited_points = num_visited_points + 1
            current_point = next_points_to_visit.pop(0)
            is_in_next_points_to_visit[current_point] = 0
            is_point_visited[current_point] = 1
        print("found a shape with: " + str(len(shape)) + " points in it.")
        
        shapes.append(shape)
    
    time_to_complete = time.process_time() - start_time
    print("total time to segment point cloud: " + str(time_to_complete) + " seconds")
    print('detected ' + str(len(shapes)) + " shapes")
    
    shapes = utilities.sort_shapes_by_size(shapes)
    
    wo = geomproc.write_options()
    wo.write_vertex_colors = True
    wo.write_face_normals = False
    file_name_no_extension, extension = os.path.splitext(file_name)
    file_name_no_extension_no_path = os.path.basename(file_name_no_extension)
    
    dir_to_file = os.path.dirname(file_name)
    dir_to_save_segmented_shapes = os.path.join(dir_to_file, file_name_no_extension_no_path + ' segmented shapes')
    if not os.path.exists(dir_to_save_segmented_shapes):
        os.mkdir(dir_to_save_segmented_shapes)
    
    for i in range(len(shapes)):
        if i > max_num_shapes_to_save:
            break
        
        mask = np.zeros(pc.point.shape[0])
        for point in shapes[i]:
            mask[point] = 1
        
        segmented_pc = geomproc.create_points(pc.point[mask.astype(bool)], radius=0, color = [0,1,0])
        # Combine everything together
        result = geomproc.mesh()
        result.append(segmented_pc)
        
        # Save the mesh
        result.save(os.path.join(dir_to_save_segmented_shapes, file_name_no_extension_no_path + ' segmented ' + str(i) + extension), wo)
    return time_to_complete
    
#amount to sample the point cloud by
sample_size = 0.3
#number of k-nearest neighbors for each point in the region growing algorithm
num_neighbours = 45
#the maximum number of shapes that will be saved as obj files
max_num_shapes_to_save = 20
file_to_segment_points_in = 'table_1_sampled_split.obj'
file_to_segment_points_in = '2 chairs and a table 2 not detected.obj'
file_to_segment_points_in = 'simple_shelf_scan_improved not detected.obj'
#utilities.split_point_cloud_at_x(pc, x_value_to_split_on = 0, split_distance = 20)
# file_to_detect_planes_in = 'table_1.obj'
# mesh = geomproc.load(file_to_detect_planes_in)
# pc = mesh.sample(5000)
pc =  geomproc.load(file_to_segment_points_in)

segment_points(pc, file_to_segment_points_in, num_neighbours, sample_size, max_num_shapes_to_save)

