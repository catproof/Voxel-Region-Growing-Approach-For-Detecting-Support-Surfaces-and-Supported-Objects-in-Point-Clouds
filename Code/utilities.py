import geomproc

import math

import random

# Import numpy library for handling matrices
import numpy as np

def swap_y_z_coordinates(pc):
    for i in range(pc.point.shape[0]):
        temp = pc.point[i][1]
        pc.point[i][1] = pc.point[i][2]
        pc.point[i][2] = temp

def split_point_cloud_at_x(pc, x_value_to_split_on = 0, split_distance = 20):
    for i in range(pc.point.shape[0]):
        if pc.point[i][0] < 0:
            pc.point[i][0] = pc.point[i][0]  - split_distance
            
def trim_point_cloud(pc, min_x = -5, max_x = 5, min_y = -2.2, max_y = 2.2, min_z = -math.inf, max_z = math.inf):
    values_to_remove = []
    for i in range(pc.point.shape[0]):
        if pc.point[i][0] < min_x:
            values_to_remove.append(i)
        elif pc.point[i][0] > max_x:
            values_to_remove.append(i)
        elif pc.point[i][1] < min_y:
            values_to_remove.append(i)
        elif pc.point[i][1] > max_y:
            values_to_remove.append(i)
        elif pc.point[i][2] < min_z:
            values_to_remove.append(i)
        elif pc.point[i][2] > max_z:
            values_to_remove.append(i)
            
    pc.point = np.delete(pc.point, values_to_remove, axis=0)
    
def remove_voxel_from_point_cloud(pc, min_x = 10, max_x = 10, min_y = 10, max_y = 10, min_z = -math.inf, max_z = math.inf):
    values_to_remove = []
    for i in range(pc.point.shape[0]):
        if pc.point[i][0] > min_x:
            values_to_remove.append(i)
        elif pc.point[i][0] < max_x:
            values_to_remove.append(i)
        elif pc.point[i][1] > min_y:
            values_to_remove.append(i)
        elif pc.point[i][1] < max_y:
            values_to_remove.append(i)
        elif pc.point[i][2] > min_z:
            values_to_remove.append(i)
        elif pc.point[i][2] < max_z:
            values_to_remove.append(i)
            
    pc.point = np.delete(pc.point, values_to_remove, axis=0)
    
def sample_mesh_and_save(mesh, file_name_to_save_as, num_samples = 5000):
    pc = mesh.sample(num_samples)
    pc = geomproc.create_points(pc.point, radius=0.00)
    result = geomproc.mesh()
    result.append(pc)
    
    wo = geomproc.write_options()
    wo.write_vertex_colors = True
    wo.write_face_normals = False
    result.save(file_name_to_save_as, wo)
    
def sample_point_cloud(pc, num_samples):
    mask = [1] * num_samples + [0] * (pc.point.shape[0] - num_samples)
    random.shuffle(mask)
    mask = np.array(mask)
    return pc.point[mask.astype(bool)]

def sort_shapes_by_size(shapes):
    return sorted(shapes, key=len, reverse=True)