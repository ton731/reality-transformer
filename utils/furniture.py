import numpy as np


def box():
    # Generate random size for the box
    size = np.random.uniform(low=0.2, high=1, size=(3,))
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

    return vertices, edges



def chair():
    # Generate random size for the box
    size = np.random.uniform(low=0.2, high=1, size=(3,))
    size_x = np.random.uniform(low=0.2, high=1, size=(1,))
    size_y = np.random.uniform(low=0.2, high=1, size=(1,))
    size_z = np.random.uniform(low=0.5, high=3, size=(1,))
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

    random_x_loc, random_y_loc = np.random.randint(low=0, high=10, size=(2,))
    vertices[:, 0] += random_x_loc
    vertices[:, 1] += random_y_loc

    # Scale the vertices by the random size
    vertices = vertices * size

    # Define the edges of the box
    edges = [
        # [0, 1],
        [1, 2], [2, 3], 
        # [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        # [0, 4],
        [1, 5], [2, 6], [3, 7],

        # [0, 11],
        [1, 10], [2, 8], [3, 9]
    ]

    return vertices, edges




def table():
    # Generate random size for the box
    size_x = np.random.uniform(low=0.5, high=2.0)
    size_y = np.random.uniform(low=0.5, high=2.0)
    size_z = np.random.uniform(low=0.5, high=max(size_x, size_y)*2)
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

    random_x_loc, random_y_loc = np.random.randint(low=0, high=10, size=(2,))
    vertices[:, 0] += random_x_loc
    vertices[:, 1] += random_y_loc

    # Scale the vertices by the random size
    vertices = vertices * size

    # Define the edges of the box
    edges = [
        # [0, 1],
        [1, 2], [2, 3], 
        # [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        # [0, 4],
        [1, 5], [2, 6], [3, 7],

        # [0, 11],
        [1, 10], [2, 8], [3, 9]
    ]

    return vertices, edges



def chandelier():

    random_height = np.random.uniform(low=25.0, high=30.0)
    rope_length = np.random.uniform(low=2.0, high=7.0)
    cube_length = np.random.uniform(low=0.1, high=0.2)
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
        # [0, 1],
        [1, 2], [2, 3], 
        # [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        # [0, 4],
        [1, 5], [2, 6], [3, 7],
        [8, 9],
    ]

    return vertices, edges

