import numpy as np
from copy import deepcopy
import cv2 as cv

'''
The stimuli information of the original stimuli is as follows:
Maximum duration: 2000ms ->
Fresh rate: 75Hz ->150 frame
Diameter: 5.0 deg -> 125 pixel
Dot speed: 5.0 deg/s -> 5 pixel /3 frame
Dot size: 0.1 deg -> 3 pixel
groups: 3
Density: 16.7 dot/deg^2/s -> 3.5627e-04/pixel^2/frame -> 6 dots over the square(rather than circle)
Viewing distance: 57cm
Monitor size: 20 inches
Monitor Resolution: 1024 by 768
'''


def RDM_generator(direction, coherence, groups=3, size=126,
                  nDot=6, dotSize=3, speed=5, frames=150):
    assert isinstance(groups, int)
    assert isinstance(size, int)
    assert isinstance(dotSize, int)
    assert isinstance(frames, int)
    assert frames % groups == 0
    assert size % 2 == 0
    coherence = coherence / 100

    circle_mask = np.zeros((size, size), dtype=np.uint8)
    cv.circle(circle_mask, (size // 2, size // 2),
              size // 2, (255, 255, 255), thickness=-1)
    nDot = int(nDot)
    pic_list = [list() for _ in range(groups)]

    for i_group in range(groups):
        dot_position = np.random.rand(nDot, 2) * size
        for i_frame in range(frames // groups):
            coh_dot_index = np.random.rand(nDot) < coherence
            N_random_dot = np.sum(~coh_dot_index)

            dot_position[coh_dot_index, 0] += speed * np.cos(np.deg2rad(direction + 90))
            dot_position[coh_dot_index, 1] += speed * np.sin(np.deg2rad(direction + 90))

            if N_random_dot > 0:
                dot_position[~coh_dot_index, :] = np.random.rand(N_random_dot, 2) * size

            dot_position[:, 0] %= size
            dot_position[:, 1] %= size

            dot_position_index = np.floor(deepcopy(dot_position)).astype(int)
            pic_temp = np.zeros((size, size), dtype=np.uint8)
            pic_temp[tuple(dot_position_index.T)] = 255
            se = cv.getStructuringElement(cv.MORPH_RECT, (dotSize, dotSize))
            pic_temp = cv.dilate(pic_temp, se)
            pic_temp = cv.bitwise_and(pic_temp, pic_temp, mask=circle_mask)

            pic_list[i_group].append(deepcopy(pic_temp))

    RDM = np.array([val for tup in zip(*pic_list) for val in tup], dtype=np.float32)

    return RDM


if __name__ == '__main__':
    RDM = RDM_generator(0, 5)

    RDM = np.tile(RDM, (3, 1, 1, 1))
    out = cv.VideoWriter('RDM.mp4', cv.VideoWriter_fourcc(*'mp4v'), 75, RDM.shape[2:], True)
    for t in range(RDM.shape[1]):
        out.write(RDM.transpose((2, 3, 0, 1))[:, :, :, t].astype(np.uint8))
    out.release()
