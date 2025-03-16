
import cv2
import numpy as np



# a = np.random.randint(0, 4, size=(1, 2, 2))
# print(a)


# mask = np.asarray([[0, 0, 0, 1, 0, 1],
#                    [0, 0, 0, 1, 1, 1],
#                    [0, 0, 0, 1, 0, 1],
#                    [0, 1, 0, 0, 1, 1],
#                    [0, 0, 0, 0, 0, 0],
#                    [1, 0, 0, 0, 0, 0]])
# kernel = np.ones((3, 3), np.uint8)
# mask = cv2.dilate(mask*1.0, kernel, iterations=1)
# print(mask)

for i in range(5):
    for j in range(i+1, 5):
        print(i, j)