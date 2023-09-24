"""
Usage:
  sparse_neural_selection.py [--total_indices=<total>] [--active_indices=<active>] [--vid_filename=<filename>]

Options:
  -h --help              Show this help message and exit.
  --total_indices=<total>  Total indices [default: 8000]
  --active_indices=<active>  Active indices [default: 3]
  --vid_filename=<filename>  Video filename
"""

import numpy as np
import random
import cv2
from displayarray import display
from docopt import docopt


def generate_random_sparse_vector(total_indices, active_indices):
    """
    Generate a random sparse vector with total dimensions and active/total sparsity.

    Args:
        total_indices (int): Total number of indices.
        active_indices (int): Number of active indices.

    Returns:
        tuple: A tuple containing unique integers, random floats, and a phase shift.
    """
    assert (
        active_indices * 2 <= total_indices
    ), "Number of active indices cannot be more than half total indices."

    unique_integers = random.sample(range(total_indices), active_indices)

    random_floats = [random.uniform(0, 1) for _ in range(active_indices)]
    random_floats = [r / sum(random_floats) for r in random_floats]

    phase_shift = random.uniform(0, 1)

    return unique_integers, random_floats, phase_shift


def reset_sparse_vectors(sparse_vectors, g_total_ind, g_active_ind):
    """
    Reset sparse vectors when at the end of the dimensions so the trained set of neurons isn't repeating.

    Args:
        sparse_vectors (list): List of sparse vectors.
        g_total_ind (int): Total number of indices.
        g_active_ind (int): Number of active indices.

    Returns:
        list: Updated list of sparse vectors.
    """
    sparse_vectors_old = sparse_vectors[-1]
    sparse_vectors = [sparse_vectors_old]
    for i in range(len(sparse_vectors), g_total_ind):
        sparse_vectors.append(generate_random_sparse_vector(g_total_ind, g_active_ind))
    return sparse_vectors


def update_image(arr, sparse_vectors, i, g_active_ind):
    """
    Update the image based on sparse vectors.

    Args:
        arr (numpy.ndarray): Image array.
        sparse_vectors (list): List of sparse vectors.
        i (float): Vector dimension index. Actually a float index. Used to scan between dimensions.
        g_active_ind (int): Number of active indices.

    Returns:
        None
    """
    for k in range(arr.shape[0] * arr.shape[1]):
        ch_1 = int(np.floor(i + sparse_vectors[k][2]))
        ch_2 = int(np.ceil(i + sparse_vectors[k][2]))
        i_mod = 1.0 - ((i + sparse_vectors[k][2]) % 1.0)

        if ch_1 in sparse_vectors[k][0][0:g_active_ind]:
            index = sparse_vectors[k][0].index(ch_1)
            arr[k // arr.shape[0]][k % arr.shape[1]] = sparse_vectors[k][1][index] * i_mod

        if ch_2 in sparse_vectors[k][0][0:g_active_ind]:
            index = sparse_vectors[k][0].index(ch_2)
            arr[k // arr.shape[0]][k % arr.shape[1]] = sparse_vectors[k][1][index] * (
                1.0 - i_mod
            )


def sparse_neural_selection_example(g_total_ind=8000, g_active_ind=3, write_vid=None):
    """
    Sparse neural selection example.

    Args:
        g_total_ind (int): Total number of indices.
        g_active_ind (int): Number of active indices.
        write_vid (Optional[str]): Video filename.

    Returns:
        None
    """
    assert g_total_ind > 2

    arr = np.zeros((200, 200))
    sparse_vectors = [
        generate_random_sparse_vector(g_total_ind, g_active_ind)
        for _ in range(arr.shape[0] * arr.shape[1])
    ]

    i = 0.05
    if write_vid is not None:
        out = cv2.VideoWriter(
            write_vid, cv2.VideoWriter_fourcc(*"mp4v"), 60.0, (200, 200)
        )

    with display(arr) as displayer:
        while displayer:
            arr[:] = 0
            if i >= g_total_ind - 2:
                sparse_vectors = reset_sparse_vectors(sparse_vectors, g_total_ind, g_active_ind)
                i = 0

            update_image(arr, sparse_vectors, i, g_active_ind)

            if write_vid is not None:
                out.write(
                    (np.repeat(arr[:, :, np.newaxis], 3, axis=2) * 255).astype(np.uint8)
                )
            i += 0.1

    if write_vid is not None:
        out.release()


if __name__ == "__main__":
    arguments = docopt(__doc__)

    total_indices = int(arguments["--total_indices"])
    active_indices = int(arguments["--active_indices"])
    vid_filename = arguments["--vid_filename"]

    sparse_neural_selection_example(total_indices, active_indices, vid_filename)
