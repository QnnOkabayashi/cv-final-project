# This file finds the key frames in a panoramic video to make a panorama

import cv2
import numpy as np
import sys
from stitcher import detectAndDescribe
from typing import Union

# This function converts a .mp4 video file into a 4D numpy array of RGB images
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


# This function returns how far across the screen objects are translated between two images
# Returns a number between 0 and 1
def get_translation_ratio(first, test, ratio: float = 0.95) -> Union[None, float]:
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

    total_translation = 0.
    for trainIdx, queryIdx in matches:
        # draw the match
        p1 = np.array([int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1])])
        p2 = np.array([int(kpsB[trainIdx][0]), int(kpsB[trainIdx][1])])
        total_translation += np.linalg.norm(p1-p2)

    avg_translation = total_translation / len(matches)

    width = first.shape[1]
    translation_ratio = avg_translation / width

    return translation_ratio


# This function returns a reduced number of images from a panoramic video that can be stitched together
# It first finds the frame interval for finding the first two frames that will stitch together nicely,
# and then uses the same frame interval between each successive image
# This means that the video must pan at a relatively constant speed, or quality will decline
def get_key_frames(filename: str, start: int = 20, count: int = 5, debug: bool = False) -> Union[None, np.array]:
    frames = get_frames(filename)
    frame_count, height, width = frames.shape[:3]

    # find number of frames to get good panning distance
    # Assume for now that there are enough frames in the video
    first_frame = frames[start]
    TOL = .1
    TARGET = .5
    interval = 20
    delta = 10
    i = 0
    while True:
        test_frame = frames[start + interval]
        translation_ratio = get_translation_ratio(first_frame, test_frame)
        if translation_ratio is None:
            # need to go back more
            interval -= delta
            continue

        if translation_ratio > TARGET + TOL:
            interval -= delta
        elif translation_ratio < TARGET - TOL:
            interval += delta
        else:
            if debug:
                print(f"Interval: {interval}")
            break

        if interval < 0:
            # Could not find any matches
            return None

        if debug:
            print(f"{i}: Interval currently at: {interval}")
        i += 1

    # get frames at that interval
    key_frames = frames[start::interval][:count]

    return key_frames


def main():
    import glob
    import os
    # if len(sys.argv) != 2:
    #     print(f"usage: python {sys.argv[0]} dataset")
    #     print(f"Example: if the dataset is `videos/mountains2.mp4`, do: python {sys.argv[0]} mountains2")
    #     sys.exit(1)

    # dataset_name = sys.argv[1]
    dataset_name = 'mountains2'
    filename = f'videos/{dataset_name}.mp4'
    print(f"Using dataset '{dataset_name}'")

    print("Finding key frames...")
    key_frames = get_key_frames(filename, count=4, debug=True)

    if key_frames is None:
        print("Could not find related frames")
        sys.exit(1)

    # write as a data set
    dir_name = os.path.join("panos", dataset_name.title())
    if not os.path.isdir(dir_name):
        # make the directory if it doesn't already exist
        os.mkdir(dir_name)
    else:
        response = input("Override existing dataset? [y/n] ")
        if response.lower() != 'y':
            print("Did not write new data set")
            sys.exit(0)
        print("Writing new data set")
        old_images = glob.glob(os.path.join(dir_name, "*.jpg"))
        for img in old_images:
            os.remove(img)

    for i, frame in enumerate(key_frames):
        cv2.imwrite(os.path.join(dir_name,
                    f"{dataset_name}_{i}.jpg"), frame)


if __name__ == '__main__':
    main()
