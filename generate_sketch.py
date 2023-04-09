# 2 point perspective
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
from tqdm import tqdm

from utils.furniture import *



def create_env():
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()

    # Define the vertices of the box
    vertices = np.array([
        [0, 0, 0],
        [0, 0, 10],
        [10, 0, 0],
        [0, 10, 0],
        [10, 0, 20],
        [0, 10, 30],
    ])

    edges = [
        [0, 1], [0, 2], [0, 3], [1, 4], [1, 5],
    ]

    # Plot the vertices and edges of the box
    # for i, vertex in enumerate(vertices):
    #     ax.scatter(vertex[0], vertex[1], vertex[2], c='r', marker='o')
        # ax.text(vertex[0], vertex[1], vertex[2], f'({i})', color='black', fontsize=8)

    for edge in edges:
        ax.plot(
            [vertices[edge[0]][0], vertices[edge[1]][0]],
            [vertices[edge[0]][1], vertices[edge[1]][1]],
            [vertices[edge[0]][2], vertices[edge[1]][2]],
            c='k'
        )
    
    return fig, ax



def draw(ax, furniture_name="chair", furniture_num=1):

    for _ in range(furniture_num):

        if furniture_name == "box":
            vertices, edges = box()
        elif furniture_name == "chair":
            vertices, edges = chair()
        elif furniture_name == "table":
            vertices, edges = table()
        elif furniture_name == "chandelier":
            vertices, edges = chandelier()
        else:
            raise ValueError(f"NO existing furniture of {furniture_name}")

        for edge in edges:
            ax.plot(
                [vertices[edge[0]][0], vertices[edge[1]][0]],
                [vertices[edge[0]][1], vertices[edge[1]][1]],
                [vertices[edge[0]][2], vertices[edge[1]][2]],
                c='k'
            )
    
    return ax


def generate_one_interior_space(save_path=None):

    fig, ax = create_env()

    # random initialize furniture number
    chair_num = random.randint(0, 3)
    table_num = random.randint(1, 3)
    chandelier_num = random.randint(1, 4)

    # draw
    ax = draw(ax, "chair", chair_num)
    ax = draw(ax, "table", table_num)
    ax = draw(ax, "chandelier", chandelier_num)

    # Set the viewing angle
    elev = random.randint(15, 20)
    azim = random.randint(40, 50)
    dist = random.randint(3, 8)
    ax.view_init(elev=elev, azim=azim)
    ax.dist = dist

    # Show the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    data_num = 3000
    save_images = True
    save_dir = Path("data/sketch2render/sketch_images")
    save_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(data_num)):
        save_path = save_dir / f"2point_{i}.png" if save_images else None
        generate_one_interior_space(save_path)