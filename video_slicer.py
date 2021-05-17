# This file finds the key frames in a panoramic video to make a panorama

import cv2
import numpy as np
from stitcher import detectAndDescribe

# Credits:
# https://stackoverflow.com/a/42166299/12401179


def get_frames(filename: str) -> np.array:
    cap = cv2.VideoCapture(filename)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frame_count, frame_height, frame_width, 3),
                   dtype='uint8')

    fc = 0
    ret = True

    while (fc < frame_count and ret):
        ret, buf[fc] = cap.read()
        fc += 1

    cap.release()

    return buf


def get_keypoint_distance(first, test, ratio=0.95) -> Union[None, float]:
    # local invariant descriptors from them
    (kpsA, featuresA) = detectAndDescribe(first)
    (kpsB, featuresB) = detectAndDescribe(test)
    # match features between the two images
    # compute the raw matches and initialize the list of actual matches
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []
    # loop over the raw matches
    for m in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe’s ratio test)
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    
    if len(matches) < 4:
        return None

    print(matches)


def get_key_frames(filename: str, start: int = 20, count: int = 5) -> np.array:
    frames = get_frames(filename)
    frame_count, height, width = frames.shape[:3]

    # find number of frames to get good panning distance
    # Assume for now that there are enough frames in the video
    first_frame = frames[start]
    interval = 20
    done = False
    while not done:
        test_frame = frames[start + interval]


    # get frames at that interval

    stop = start + (count * interval)
    key_frames = frames[start:stop:interval]
    return key_frames


def main():
    key_frames = get_key_frames('pano_frames.mp4')


if __name__ == '__main__':
    main()
