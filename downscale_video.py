##############################################################################
#
# File: downscale_video.py
# Authors: Quinn Okabayashi, Theron Mansilla
# Course: ENGR 27
# Date: May 18 2021
#
# Resources:
# * Writing video to a file
# * https://stackoverflow.com/a/54731615/12401179
#
##############################################################################
#
# This file downscales videos to 800x600 to make it easier to use for
# panoramic stitching, as well as reduce size of the overall repo.
#
##############################################################################

import sys
import os
import cv2
import numpy as np
from video_slicer import get_frames


def rescale_video(video: np.array, output_name: str):
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(os.path.join(
        'videos', f'{output_name}.mp4'), fourcc, 20.0, (800, 600))

    for frame in video:
        resized_frame = cv2.resize(frame, (800, 600))
        out.write(resized_frame)

    out.release()
    cv2.destroyAllWindows()


def main():
    if len(sys.argv) != 3:
        print("usage: python downscale_video.py VIDEOPATH NEWNAME")
        sys.exit(0)

    filename = sys.argv[1]
    output_name = sys.argv[2]

    video = get_frames(filename)
    rescale_video(video, output_name)


if __name__ == '__main__':
    main()
