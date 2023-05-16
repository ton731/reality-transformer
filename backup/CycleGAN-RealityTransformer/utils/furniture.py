import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random



def assign_random_location(interior_space, vertices):

    # first move the furniture to the first quadrant
    if np.min(vertices[:, 0]) < 0:
        vertices[:, 0] -= np.min(vertices[:, 0])
    if np.min(vertices[:, 1]) < 0:
        vertices[:, 1] -= np.min(vertices[:, 1])
    if np.min(vertices[:, 2]) < 0:
        vertices[:, 2] -= np.min(vertices[:, 2])

    # find the boundary of the furniture
    min_x, max_x = np.min(vertices[:, 0]), np.max(vertices[:, 0])
    min_y, max_y = np.min(vertices[:, 1]), np.max(vertices[:, 1])

    # generate 10 random coord and find the best location to place the furniture
    iteration = 100
    best_cover_area = np.inf
    best_x, best_y = None, None
    for i in range(iteration):
        random_x_loc, random_y_loc = np.random.randint(low=0, high=15, size=(2,))
        new_min_x, new_max_x = int(min_x + random_x_loc), int(max_x + random_x_loc)
        new_min_y, new_max_y = int(min_y + random_y_loc), int(max_y + random_y_loc)
        cover_existed_furniture_area = np.sum(interior_space[new_min_x:new_max_x+1, new_min_y:new_max_y+1])
        # print(f"iter: {i}, best_cover_area: {best_cover_area}, current_area: {cover_existed_furniture_area}")
        if cover_existed_furniture_area < best_cover_area:
            best_cover_area = cover_existed_furniture_area
            best_x, best_y = random_x_loc, random_y_loc
            if best_cover_area == 0:
                break
    
    vertices[:, 0] += best_x
    vertices[:, 1] += best_y
    new_min_x, new_max_x = int(min_x + best_x), int(max_x + best_x)
    new_min_y, new_max_y = int(min_y + best_y), int(max_y + best_y)
    # print("new boundary:", new_min_x, new_max_x, new_min_y, new_max_y)
    interior_space[new_min_x:new_max_x+1, new_min_y:new_max_y+1] = True

    return interior_space, vertices




def box(interior_space, seed):
    # Generate random size for the box
    size_x = np.random.uniform(low=0.2, high=1, size=(1,))
    size_y = np.random.uniform(low=0.2, high=1, size=(1,))
    size_z = np.random.uniform(low=0.8, high=3, size=(1,))
    size = np.concatenate([size_x, size_y, size_z])

    # Define the vertices of the box
    vertices = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ])

    random_x_loc, random_y_loc = np.random.randint(low=0, high=10, size=(2,))
    vertices[:, 0] += random_x_loc
    vertices[:, 1] += random_y_loc

    # Scale the vertices by the random size
    vertices = vertices * size

    # Define the edges of the box
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]

    # Define faces for coloring and the random color
    faces = None
    color = None

    return vertices, edges, faces, color, interior_space



def chair(interior_space, seed):
    # Generate random size for the box
    size_x = np.random.uniform(low=0.8, high=1, size=(1,))
    size_y = np.random.uniform(low=0.8, high=1, size=(1,))
    size_z = np.random.uniform(low=1.0, high=2, size=(1,))
    size = np.concatenate([size_x, size_y, size_z])

    # Define the vertices of the box
    vertices = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],

        [1, 1, -2],
        [-1, 1, -2],
        [1, -1, -2],
        [-1, -1, -2],
    ])

    # Scale the vertices by the random size
    vertices = vertices * size

    # assign good random location
    interior_space, vertices = assign_random_location(interior_space, vertices)

    # Define the edges of the box
    edges = [
        [0, 1],
        [1, 2], [2, 3], 
        [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4],
        [1, 5], [2, 6], [3, 7],
        [0, 11],
        [1, 10], [2, 8], [3, 9]
    ]

    # Define faces for coloring and the random color
    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 4, 7, 3],
        [1, 2, 6, 5],
        [0, 1, 5, 4],
        [2, 3, 7, 6]
    ]
    cmap = plt.get_cmap('pink')
    random.seed(seed)
    rand_num = random.uniform(0.05, 1)
    color = cmap(rand_num)
    color = colors.to_rgb(color)

    return vertices, edges, faces, color, interior_space



def chair_seatback(interior_space, seed):
    # Generate random size for the box
    size_x = np.random.uniform(low=0.5, high=0.8, size=(1,))
    size_y = np.random.uniform(low=0.5, high=0.8, size=(1,))
    size_z = np.random.uniform(low=0.7, high=2, size=(1,))
    size = np.concatenate([size_x, size_y, size_z])

    # Define the vertices of the box
    vertices = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 0],
        [1, -1, 0],
        [1, 1, 0],
        [-1, 1, 0],

        [1, 1, -2],
        [-1, 1, -2],
        [1, -1, -2],
        [-1, -1, -2],
        
        [-1, -1, 2],
        [1, -1, 2],
    ])

    # Scale the vertices by the random size
    vertices = vertices * size

    # assign good random location
    interior_space, vertices = assign_random_location(interior_space, vertices)

    # Define the edges of the box
    edges = [
        [0, 1],[1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],

        [0, 11], [1, 10], [2, 8], [3, 9],
        
        [12, 4],[13, 5],[12, 13]
    ]

    # Define faces for coloring and the random color
    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [0, 4, 7, 3],
        [4, 5, 13, 12]
    ]
    cmap = plt.get_cmap('RdGy')
    random.seed(seed)
    rand_num = random.uniform(0.05, 1)
    color = cmap(rand_num)
    color = colors.to_rgb(color)
    
    return vertices, edges, faces, color, interior_space



def sofa(interior_space, seed):
    # Generate random size for the box
    size_x = np.random.uniform(low=1.0, high=2.0, size=(1,))
    size_y = np.random.uniform(low=1.0, high=2.0, size=(1,))
    size_z = np.random.uniform(low=1.0, high=1.8, size=(1,))
    size = np.concatenate([size_x, size_y, size_z])
    
    width = np.random.randint(2,8)

    # Define the vertices of the box
    vertices = np.array([
        [-1, -1, -1],
        [-1+width, -1, -1],
        [-1+width, 1, -1],
        [-1, 1, -1],
        [-1, -1, 0],
        [-1+width, -1, 0],
        [-1+width, 1, 0],
        [-1, 1, 0],

        [-1+width, 1, -2],
        [-1, 1, -2],
        [-1+width, -1, -2],
        [-1, -1, -2],
        
        [-1, -1, 2],
        [-1+width, -1, 2],
        
        [-1, -1, 1],
        [-1+width, -1, 1],
        [-1+width, 1, 1],
        [-1, 1, 1],
    ])

    # Scale the vertices by the random size
    vertices = vertices * size

    # assign good random location
    interior_space, vertices = assign_random_location(interior_space, vertices)

    # Define the edges of the box
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],

        [0, 11], [1, 10], [2, 8], [3, 9],
        
        [12, 4], [13, 5], [12, 13],
        
        [15, 16], [16, 6] , [17, 14], [17, 7]
    ]

    # Define faces for coloring and the random color
    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [0, 4, 7, 3],
        [4, 5, 13, 12],
        [4, 7, 17, 14],
        [5, 6, 16, 15]
    ]
    cmap = plt.get_cmap('RdGy')
    random.seed(seed)
    rand_num = random.uniform(0.05, 1)
    color = cmap(rand_num)
    color = colors.to_rgb(color)

    return vertices, edges, faces, color, interior_space




def cabinet(interior_space, seed):
    # Generate random size for the box
    size_x = np.random.uniform(low=0.6, high=1.2, size=(1,))
    size_y = np.random.uniform(low=0.6, high=1.2, size=(1,))
    size_z = np.random.uniform(low=1.0, high=3, size=(1,))
    size = np.concatenate([size_x, size_y, size_z])
    
    grid_number = np.array([np.random.randint(1,6), np.random.randint(1,6)])
    height = grid_number[0]
    width = grid_number[1]
    # Define the vertices of the box
    vertices = np.array([
        [-1, -1, -1],
        [-1+width, -1, -1],
        [-1+width, 1, -1],
        [-1, 1, -1],
        [-1, -1, -1+height],
        [-1+width, -1, -1+height],
        [-1+width, 1, -1+height],
        [-1, 1, -1+height],
    ])
    
    # Define the edges and faces of the box
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        # [2, 3, 7, 6],
        [0, 3, 7, 4]
    ]
    
    for i in range(height):
        if 0<i<grid_number[0]:
            layer = np.array([
                [-1, -1, -1+i],
                [-1+width, -1, -1+i],
                [-1+width, 1, -1+i],
                [-1, 1, -1+i]
                ])
            vertices = np.concatenate((vertices,layer),axis=0)
            edges.extend([
                [7+4*(i-1)+1, 7+4*(i-1)+2],
                [7+4*(i-1)+2, 7+4*(i-1)+3],
                [7+4*(i-1)+3, 7+4*(i-1)+4],
                [7+4*(i-1)+4, 7+4*(i-1)+1]
                ])
            faces.append([7+4*(i-1)+1, 7+4*(i-1)+2, 7+4*(i-1)+3, 7+4*(i-1)+4])
    
    for j in range(width):
        if 0<j<grid_number[1]:
            add_grid = np.array([
                [-1+j, -1, -1],
                [-1+j, 1, -1],
                [-1+j, 1, -1+height],
                [-1+j, -1, -1+height]
                ])
            vertices = np.concatenate((vertices,add_grid),axis=0)
            edges.append([7+4*(height-1)+4*(j-1)+2,7+4*(height-1)+4*(j-1)+3])
            faces.append([7+4*(height-1)+4*(j-1)+1, 7+4*(height-1)+4*(j-1)+2, 7+4*(height-1)+4*(j-1)+3, 7+4*(height-1)+4*(j-1)+4])

    # Scale the vertices by the random size
    vertices = vertices * size

    # assign good random location
    interior_space, vertices = assign_random_location(interior_space, vertices)

    # Define the random color
    cmap = plt.get_cmap('RdGy')
    random.seed(seed)
    rand_num = random.uniform(0.05, 1)
    color = cmap(rand_num)
    color = colors.to_rgb(color)
    
    return vertices, edges, faces, color, interior_space



def table(interior_space, seed):
    # Generate random size for the box
    size_x = np.random.uniform(low=0.8, high=2.0)
    size_y = np.random.uniform(low=0.8, high=2.0)
    size_z = np.random.uniform(low=0.8, high=max(size_x, size_y)*1.5)
    size = np.array([size_x, size_y, size_z])
    # size = np.array([1, 1, 1])

    table_thickness = np.random.uniform(low=0.4, high=1.0)
    bottom_z = np.random.uniform(low=-2.0, high=0.0)
    leg_height = np.random.uniform(low=1.0, high=3.0)

    # Define the vertices of the box
    vertices = np.array([
        [-1, -1, bottom_z + leg_height],
        [1, -1, bottom_z + leg_height],
        [1, 1, bottom_z + leg_height],
        [-1, 1, bottom_z + leg_height],
        [-1, -1, bottom_z + leg_height + table_thickness],
        [1, -1, bottom_z + leg_height + table_thickness],
        [1, 1, bottom_z + leg_height + table_thickness],
        [-1, 1, bottom_z + leg_height + table_thickness],

        [1, 1, bottom_z],
        [-1, 1, bottom_z],
        [1, -1, bottom_z],
        [-1, -1, bottom_z],
    ])

    # Scale the vertices by the random size
    vertices = vertices * size

    # assign good random location
    interior_space, vertices = assign_random_location(interior_space, vertices)

    # Define the edges of the box
    edges = [
        [0, 1],
        [1, 2], [2, 3], 
        [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4],
        [1, 5], [2, 6], [3, 7],

        [0, 11],
        [1, 10], [2, 8], [3, 9]
    ]

    # Define faces for coloring and the random color
    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 4, 7, 3],
        [1, 2, 6, 5],
        [0, 1, 5, 4],
        [2, 3, 7, 6]
    ]
    cmap = plt.get_cmap('copper')
    random.seed(seed)
    rand_num = random.uniform(0.05, 1)
    color = cmap(rand_num)
    color = colors.to_rgb(color)

    return vertices, edges, faces, color, interior_space



def chandelier(interior_space, seed):

    random_height = np.random.uniform(low=25.0, high=30.0)
    rope_length = np.random.uniform(low=2.0, high=7.0)
    cube_length = np.random.uniform(low=0.15, high=0.3)
    cube_height = np.random.uniform(low=0.5, high=2.0)

    # Define the vertices of the box
    vertices = np.array([
        [-cube_length, -cube_length, random_height - rope_length - cube_height],
        [cube_length, -cube_length, random_height - rope_length - cube_height],
        [cube_length, cube_length, random_height - rope_length - cube_height],
        [-cube_length, cube_length, random_height - rope_length - cube_height],
        [-cube_length, -cube_length, random_height - rope_length],
        [cube_length, -cube_length, random_height - rope_length],
        [cube_length, cube_length, random_height - rope_length],
        [-cube_length, cube_length, random_height - rope_length],

        # cube center
        [0, 0, random_height - rope_length],
        # top center
        [0, 0, random_height],
    ])

    random_x_loc, random_y_loc = np.random.randint(low=0, high=10, size=(2,))
    vertices[:, 0] += random_x_loc
    vertices[:, 1] += random_y_loc

    # Define the edges of the box
    edges = [
        [0, 1],
        [1, 2], [2, 3], 
        [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4],
        [1, 5], [2, 6], [3, 7],
        [8, 9],
    ]

    # Define faces for coloring and the random color
    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 4, 7, 3],
        [1, 2, 6, 5],
        [0, 1, 5, 4],
        [2, 3, 7, 6]
    ]
    cmap = plt.get_cmap('magma')
    random.seed(seed)
    rand_num = random.uniform(0.15, 1)
    color = cmap(rand_num)
    color = colors.to_rgb(color)

    return vertices, edges, faces, color, interior_space



def door(interior_space, seed):
    # 產生一個隨機大小的數組作為門的尺寸
    size_x = np.random.uniform(low=0.2, high=1, size=(1,))
    size_y = np.random.uniform(low=0.2, high=1, size=(1,))
    size_z = np.random.uniform(low=0.8, high=1, size=(1,))
    size = np.concatenate([size_x, size_y, size_z])

    # 定義門的四個頂點坐標
    vertices = np.array([
        [3, 0, 0],
        [3, 0, 5],
        [1, 0, 5],
        [1, 0, 0],
    ])

    # 按照隨機大小縮放門的頂點坐標
    vertices = vertices * size

    # 隨機移動門的位置
    random_x_loc, random_y_loc = np.random.randint(low=0, high=2, size=(2,))
    vertices[:, 0] += random_x_loc
    # vertices[:, 0] += random_y_loc

    # 定義門的邊
    edges = [
        [0, 1], [1, 2], [2, 3], 
    ]

    # Define faces for coloring and the random color
    faces = None
    color = None

    return vertices, edges, faces, color, interior_space




def window(interior_space, seed):
    # 產生一個隨機大小的數組作為門的尺寸
    size_x = np.random.uniform(low=0.2, high=1, size=(1,))
    size_y = np.random.uniform(low=0.2, high=1, size=(1,))
    size_z = np.random.uniform(low=0.8, high=1, size=(1,))
    size = np.concatenate([size_x, size_y, size_z])

    # 定義門的四個頂點坐標
    vertices = np.array([
        [3, 0, 0],
        [3, 0, 3],
        [1, 0, 3],
        [1, 0, 0],
        [1, 0, 3],
        [1, 0, 0],
    ])

    # 按照隨機大小縮放門的頂點坐標
    vertices = vertices * size

    # 隨機移動門的位置
    random_x_loc, random_y_loc = np.random.randint(low=0, high=2, size=(2,))
    vertices[:, 0] += random_x_loc
    # vertices[:, 0] += random_y_loc

    # 定義門的邊
    edges = [
        [0, 1], [1, 2], [2, 3], 
    ]

    # Define faces for coloring and the random color
    faces = None
    color = None

    return vertices, edges, faces, color, interior_space