import random
import numpy as np


def simul_transform(lr_img, hr_img, flip=True, rotation=False):
    if flip:
        flip_prob = random.uniform(0, 1)
        if flip_prob >= 0.5:
            lr_img = np.flip(lr_img, 2)
            hr_img = np.flip(hr_img, 2)

    if rotation:
        rot_times = random.randint(1, 4)

        lr_img = np.rot90(lr_img, rot_times, (1, 2))

        hr_img = np.rot90(hr_img, rot_times, (1, 2))


    return np.ascontiguousarray(lr_img), np.ascontiguousarray(hr_img)




