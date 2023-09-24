from displayarray import display
import numpy as np
import random
import cv2


# log scale training example. log_2(200x200)=15. 200*200=40000. gcd=5. 3/8000
g_total_ind = 8000
g_active_ind = 3

assert g_total_ind>2
def generate_random_sparse_vector(i, total_indices, active_indices):
    assert (active_indices * 2 <= total_indices, "Number of active indices cannot be more than half total indices.")

    unique_integers = random.sample(range(total_indices), active_indices)

    random_floats = []
    random_floats_1 = [random.uniform(0, 1) for _ in range(active_indices)]
    random_floats.extend([r / (sum(random_floats_1)) for r in random_floats_1])

    phase_shift = random.uniform(0, 1)

    return unique_integers, random_floats_1, phase_shift


arr = np.zeros((200, 200))

spvecs = []
for i in range(arr.shape[0] * arr.shape[1]):
    spvecs.append(generate_random_sparse_vector(i, g_total_ind, g_active_ind))

i = 0.05
out = cv2.VideoWriter('1.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 60.0, (200,200))
with display(arr) as displayer:
    while displayer:
        arr[:] = 0
        if i >= g_total_ind-2:
            spvecs_old = spvecs[-1]
            spvecs = [spvecs_old]
            for i in range(arr.shape[0] * arr.shape[1]):
                spvecs.append(generate_random_sparse_vector(i, g_total_ind, g_active_ind))
            i = 0
        for k in range(arr.shape[0] * arr.shape[1]):
            ch_1 = int(np.floor(i+spvecs[k][2]))
            ch_2 = int(np.ceil(i+spvecs[k][2]))
            i_mod = 1.0-((i+spvecs[k][2]) % 1.0)
            if ch_1 in spvecs[k][0][0:g_active_ind]:
                index = spvecs[k][0].index(ch_1)
                arr[k // arr.shape[0]][k % arr.shape[1]] = spvecs[k][1][index] * i_mod
            if ch_2 in spvecs[k][0][0:g_active_ind]:
                index = spvecs[k][0].index(ch_2)
                arr[k // arr.shape[0]][k % arr.shape[1]] = spvecs[k][1][index] * (1.0 - i_mod)
        out.write((np.repeat(arr[:, :, np.newaxis], 3, axis=2)*255).astype(np.uint8))
        i += .1
out.release()