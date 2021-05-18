##############################################################################
#
# File: stitcher.py
# Authors: Quinn Okabayashi, Theron Mansilla
# Course: ENGR 27
# Date: May 17 2021
#
# Resources:
# * Using ORB to find key points:
# * https://medium.com/analytics-vidhya/panorama-formation-using-image-stitching-using-opencv-1068a0e8e47b
#
# * Stitching together multiple images without common key points:
# * https://towardsdatascience.com/image-panorama-stitching-with-opencv-2402bde6b46c
#
##############################################################################
#
# This file stitches panoramas together :)
#
##############################################################################

import numpy as np
import cv2
from typing import Tuple, Union, List


def stitch(images, ratio=0.95, reprojThresh=4.0, showMatches=None) -> Union[None, np.array, Tuple[np.array, np.array]]:
    # local invariant descriptors from them
    (imageA, imageB) = images
    (keypointsA, featuresA) = detect_and_describe(imageA)
    (keypointsB, featuresB) = detect_and_describe(imageB)

    # match features between the two images
    M = match_keypoints(keypointsB, keypointsA, featuresB,
                        featuresA, ratio, reprojThresh)

    # if the match is None, then there aren’t enough matched
    if M is None:
        return None

    # otherwise, apply a perspective warp to stitch the images together
    (matches, H, status) = M
    result = cv2.warpPerspective(
        imageB, H, (imageB.shape[1] + imageA.shape[1], imageB.shape[0]))
    result[0:imageA.shape[0], 0:imageA.shape[1]] = imageA

    # check to see if the keypoint matches should be visualized
    if showMatches is not None:
        vis = draw_matches(imageB, imageA, keypointsB,
                           keypointsA, matches, status)
        # return a tuple of the stitched image and the visualization
        cv2.imshow(showMatches, vis)

    # return the stitched image
    return result


def detect_and_describe(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect keypoints in the image
    orb = cv2.ORB_create()

    kps, features = orb.detectAndCompute(image, None)
    # convert the keypoints from KeyPoint objects to NumPy arrays
    kps = np.float32([kp.pt for kp in kps])
    # return a tuple of keypoints and features
    return (kps, features)


def match_keypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
    # compute the raw matches and initialize the list of actual matches
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []     # loop over the raw matches
    for m in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe’s ratio test)
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
            # computing a homography requires at least 4 matches

    if len(matches) > 4:
        # construct the two sets of points
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
        # compute the homography between the two sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        # return the matches along with the homograpy matrix
        # and status of each matched point
        return (matches, H, status)
    # otherwise, no homograpy could be computed
    return None


def draw_matches(imageA, imageB, kpsA, kpsB, matches, status):
    # initialize the output visualization image
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB
    # loop over the matches
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successfully matched
        if s == 1:
            # draw the match
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (255, 0, 0), 3)

    # return the visualization
    return vis


def stitch_many(frames: List[np.array]) -> np.array:
    # TODO: currently using naïve implementation, make better
    # ! Matt, please help us ;_;
    stitched = frames[-1]
    for image in frames[-1::-1]:
        stitched = stitch([image, stitched])
    # stitched = frames[0]
    # for i, image in enumerate(frames[1:]):
    #     stitched = stitch([stitched, image], onto='left', showMatches=str(i))
    return stitched
