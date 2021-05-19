##############################################################################
#
# File: main.py
# Authors: Quinn Okabayashi, Theron Mansilla
# Course: ENGR 27
# Date: May 18 2021
#
##############################################################################
#
# This file brings all the modules together, allowing users to interactively
# to select existing datasets or create custom ones from videos automatically.
# Then, they can view the individual images or stitch together a panorama, 
# as well as save the results to outputs/
#
##############################################################################

import numpy as np
import sys
import os
import cv2
from typing import List, Dict
from demo import *
from stitcher import stitch


def main():
    while True:
        res: Union[None, Tuple[np.array, str]] = select_video_menu()
        if res is None:
            sys.exit(0)

        images, dataset = res

        while True:
            display_dataset_menu(dataset)
            stitched: Union[None, np.array] = None

            key = cv2.waitKey(0)

            if key == 27:
                break
            elif key == ord('1'):
                dataset_demo(images)
            elif key == ord('2'):
                if stitched is None:
                    stitched = stitch(images)
                cv2.imshow(WINDOW_NAME, stitched)
                cv2.waitKey(0)
            elif key == ord('3'):
                # save stitched image
                if stitched is None:
                    stitched = stitch(images)
                filename = f"{dataset}_{len(images)}.jpg"
                cv2.imwrite(os.path.join('outputs', filename), stitched)


if __name__ == "__main__":
    main()
