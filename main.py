##############################################################################
#
# File: main.py
# Authors: Quinn Okabayashi, Theron Mansilla
# Course: ENGR 27
# Date: May 17 2021
#
##############################################################################
#
# This file brings all the modules together, allowing users to interactively
# to select existing datasets or create custom ones from videos automatically.
# Then, they can view the individual images or stitch together a panorama.
#
##############################################################################

import numpy as np
import sys
import os
import cv2
from typing import List, Dict
from demo import *
from stitcher import *


def main():
    DATASET = None
    images: List[np.array] = []
    cached: Dict[str, np.array] = {}

    while True:
        images, DATASET = get_dataset_menu()

        while True:
            display_dataset_menu(DATASET)

            key = cv2.waitKey(0)

            if key == 27:
                break
            elif key == ord("1"):
                dataset_demo(images)
            elif key == ord("2"):
                # if DATASET not in cached.keys():
                #     cached[DATASET] = stitch_many(images)

                # result = cached[DATASET]
                result = stitch_many(images)
                cv2.imshow(WINDOW_NAME, result)
                cv2.waitKey(0)


if __name__ == "__main__":
    main()
