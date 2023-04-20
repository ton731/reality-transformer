import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
import cv2
from tqdm import tqdm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as colors

from utils.furniture import *


def create_env():
    # Create a 3D plot
    # ref for z-order: https://stackoverflow.com/questions/37611023/3d-parametric-curve-in-matplotlib-does-not-respect-zorder-workaround
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    ax.set_axis_off()

    # record the usage of the space
    interior_space = np.full((30, 30), False)

    # Define the vertices of the box
    vertices = np.array([
        [0, 0, 0],
        [0, 0, 10],
        [20, 0, 0],
        [0, 20, 0],
        [20, 0, 30],
        [0, 20, 50],

        # fake points for plotting ground
        [60, 0, 0],
        [60, 60, 0],
        [0, 60, 0],

        # fake points for plotting ceiling
        [60, 0, 50],
        [60, 60, 50],
        [0, 60, 50],
    ])

    edges = [
        [0, 1], [0, 2], [0, 3], [1, 4], [1, 5],
    ]

    for edge in edges:
        ax.plot(
            [vertices[edge[0]][0], vertices[edge[1]][0]],
            [vertices[edge[0]][1], vertices[edge[1]][1]],
            [vertices[edge[0]][2], vertices[edge[1]][2]],
            c='k'
        )
    
    return fig, ax, vertices, interior_space



def set_env_color(ax, vertices):
    # color the interior space
    cmap = plt.get_cmap('Greys')
    color_wall = colors.to_rgb(cmap(random.uniform(0, 0.5)))
    color_ground = colors.to_rgb(cmap(random.uniform(0, 0.3)))
    color_ceiling = colors.to_rgb(cmap(random.uniform(0.4, 1.0)))
    color_ceiling = (1, 0, 0)


    faces_wall = [
        [vertices[0], vertices[1], vertices[4], vertices[2]],
        [vertices[0], vertices[1], vertices[5], vertices[3]]
    ]
    faces_ground = [
        [vertices[0], vertices[2], vertices[6], vertices[7], vertices[8], vertices[3]],
    ]

    faces_ceiling = [
        [vertices[1], vertices[4], vertices[9], vertices[10], vertices[11], vertices[5]],
    ]
    
    collection_wall = Poly3DCollection(faces_wall, facecolors=color_wall, linewidths=1, edgecolors='k', alpha=1.0)
    collection_ground = Poly3DCollection(faces_ground, facecolors=color_ground, linewidths=1, edgecolors='k', alpha=1.0)
    collection_ceiling = Poly3DCollection(faces_ceiling, facecolors=color_ceiling, linewidths=1, edgecolors='k', alpha=1.0)
    collection_wall.set_zorder(2)
    collection_ground.set_zorder(2)
    collection_ceiling.set_zorder(1)
    ax.add_collection3d(collection_wall)
    ax.add_collection3d(collection_ground)
    # ax.add_collection3d(collection_ceiling)

    return ax



def draw(ax, interior_space, furniture_name="chair", furniture_num=1):

    seed = np.random.randint(0, 1000)

    for _ in range(furniture_num):

        if furniture_name == "box":
            vertices, edges, faces, color, interior_space = box(interior_space, seed)
        elif furniture_name == "chair":
            vertices, edges, faces, color, interior_space = chair(interior_space, seed)
        elif furniture_name == "table":
            vertices, edges, faces, color, interior_space = table(interior_space, seed)
        elif furniture_name == "chandelier":
            vertices, edges, faces, color, interior_space = chandelier(interior_space, seed)
        elif furniture_name == "cabinet":
            vertices, edges, faces, color, interior_space = cabinet(interior_space, seed)
        elif furniture_name == "chair_seatback":
            vertices, edges, faces, color, interior_space = chair_seatback(interior_space, seed)
        elif furniture_name == "sofa":
            vertices, edges, faces, color, interior_space = sofa(interior_space, seed)
        elif furniture_name == "door":
            vertices, edges, faces, color, interior_space = door(interior_space, seed)
        else:
            raise ValueError(f"NO existing furniture of {furniture_name}")

        # plot the lines
        for edge in edges:
            ax.plot(
                [vertices[edge[0]][0], vertices[edge[1]][0]],
                [vertices[edge[0]][1], vertices[edge[1]][1]],
                [vertices[edge[0]][2], vertices[edge[1]][2]],
                c='k'
            )
        
        # plot the faces
        if faces:
            faces = [[vertices[i] for i in face] for face in faces]
            collection = Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='k', alpha=1.0)
            collection.set_zorder(10)
            ax.add_collection3d(collection)

    
    return ax, interior_space




def generate_one_interior_space(save_path=None):

    fig, ax, vertices, interior_space = create_env()

    # Color the env
    ax = set_env_color(ax, vertices)

    # random initialize furniture number
    chair_num = random.randint(0, 2)
    table_num = random.randint(0, 2)
    chandelier_num = random.randint(0, 4)
    cabinet_num = random.randint(0, 2)
    chair_seatback_num = random.randint(0, 2)
    sofa_num= random.randint(0, 1)
    door_num = random.randint(0, 1)

    # draw
    # bigger furniture should be placed first
    ax, interior_space = draw(ax, interior_space, "sofa", sofa_num)
    ax, interior_space = draw(ax, interior_space, "table", table_num)
    ax, interior_space = draw(ax, interior_space, "cabinet", cabinet_num)
    ax, interior_space = draw(ax, interior_space, "chair", chair_num)
    ax, interior_space = draw(ax, interior_space, "chair_seatback", chair_seatback_num)
    ax, interior_space = draw(ax, interior_space, "chandelier", chandelier_num)
    ax, interior_space = draw(ax, interior_space, "door", door_num)

    # Set the viewing angle
    elev = random.randint(15, 20)
    azim = random.randint(40, 50)
    dist = random.randint(4, 7)
    ax.view_init(elev=elev, azim=azim)
    ax.dist = dist

    # Show the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()



def crop_view(save_path):
    view = cv2.imread(str(save_path), 1)

    object_pixels = np.argwhere(view == [0, 0, 0])

    min_x = np.min(object_pixels[:, 1])
    max_x = np.max(object_pixels[:, 1])
    min_y = np.min(object_pixels[:, 0])
    max_y = np.max(object_pixels[:, 0])

    new_view = view[min_y-2:max_y+2, min_x-2:max_x+2]

    cv2.imwrite(str(save_path), new_view)





if __name__ == "__main__":
    data_num = 10
    save_images = False
    save_dir = Path("data/sketch2render/sketch_images")
    save_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(data_num)):
        save_path = save_dir / f"2point_{i}.png" if save_images else None
        generate_one_interior_space(save_path)
        if save_images:
            crop_view(save_path)



