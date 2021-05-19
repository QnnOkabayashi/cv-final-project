##############################################################################
#
# File: stitcher.py
# Authors: Quinn Okabayashi, Theron Mansilla
# Course: ENGR 27
# Date: May 18 2021
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
# This file provides utilities for stitching together many images into 
# a single panoramic image.
#
##############################################################################

import numpy as np
import cv2
from typing import Tuple, Union, List


def stitch(images: List[np.array], base_idx: Union[None, int] = None) -> Union[None, np.array]:
    if base_idx is None:
        # set the base image to the image closest to the center
        base_idx = len(images) // 2
    elif base_idx < 0:
        return None

    # get homographies between each image
    homographies = get_homography_list(images)
    if homographies is None:
        return None

    # get homographies to center around the image at base_idx
    mapped_homographies = map_homographies(homographies, base_idx)

    def get_corners(image: np.array) -> np.array:
        H, W, *_ = image.shape
        corners = np.array([[[0, 0]],
                            [[W, 0]],
                            [[W, H]],
                            [[0, H]]],
                           dtype='float32')
        return corners

    all_corners_transformed = []

    for image, H in zip(images, mapped_homographies):
        corners = get_corners(image)
        corners = cv2.perspectiveTransform(corners, H)
        all_corners_transformed.append(corners)

    # bounding box
    points = np.vstack(all_corners_transformed)

    x0, y0, Wc, Hc = cv2.boundingRect(points)

    # warping
    T = np.array([[1,   0, -x0],
                  [0,   1, -y0],
                  [0,   0,   1]],
                 dtype='float32')

    all_warped = []
    for image, H in zip(images, mapped_homographies):
        M = T @ H
        warped = cv2.warpPerspective(image, M, dsize=(Wc, Hc))
        all_warped.append(warped)

    # count image intersections
    intersections = np.zeros_like(all_warped[0])
    for warped in all_warped:
        intersections[warped > 0] += 1

    mask = intersections.astype(bool)

    # reduce brightness on intersections
    for warped in all_warped:
        warped[mask] //= intersections[mask]

    # compose result
    final_image = np.zeros_like(all_warped[0]).astype(np.uint8)

    for warped in all_warped:
        final_image += warped

    return final_image


def map_homographies(homographies: List[np.array], base: int) -> Union[None, List[np.array]]:
    if not 0 <= base < len(homographies):
        return None

    # do the left side
    curr = np.eye(3)
    left = []
    # go from the base down to the first one
    indexed = list(enumerate(homographies))
    for i, H in reversed(indexed[:base]):
        curr = curr @ H
        left = [curr] + left

    # do the right side
    prev = np.eye(3)
    right = []
    for i, H in indexed[base:]:
        prev = prev @ np.linalg.inv(H)
        right.append(prev)

    mapped_homographies = left + [np.eye(3)] + right

    return mapped_homographies


def get_homography_list(images: List[np.array]) -> Union[None, List[np.array]]:
    orb = cv2.ORB_create()

    def get_homography(imageA: np.array, imageB: np.array) -> Union[None, np.array]:
        RATIO: float = 0.95
        REPROJ_THRESH: float = 4.0
        # REPROJ_THRESH: float = 6.0

        kpsA, featuresA = orb.detectAndCompute(imageA, None)
        kpsB, featuresB = orb.detectAndCompute(imageB, None)

        # convert the keypoints from KeyPoint objects to NumPy arrays
        kpsA = np.float32([kp.pt for kp in kpsA])
        kpsB = np.float32([kp.pt for kp in kpsB])

        # Find the matching points across the two images
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

        matches = []
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * RATIO:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            H, *_ = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, REPROJ_THRESH)

            # ! uncomment below to see the matches that ORB is finding for each image
            # draw_matches(imageA, imageB, kpsA, kpsB, matches)

            return H
        else:
            return None

    homographies = []
    for imageA, imageB in zip(images[:-1], images[1:]):
        H = get_homography(imageA, imageB)
        if H is not None:
            homographies.append(H)
        else:
            return None

    return homographies


def draw_matches(imageA, imageB, kpsA, kpsB, matches):
    # initialize the output visualization image
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype='uint8')
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB
    # loop over the matches
    for trainIdx, queryIdx in matches:
        # draw the match
        ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
        ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
        cv2.line(vis, ptA, ptB, (255, 0, 0), 1)

    # return the visualization
    cv2.imshow("Keypoints", vis)
    cv2.waitKey(0)
