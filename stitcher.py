##############################################################################
#
# File: stitcher.py
# Authors: Quinn Okabayashi, Theron Mansilla
# Course: ENGR 27
# Date: May 11 2021
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
import sys
import cv2
from typing import Tuple, Union


def stitch(images, ratio=0.95, reprojThresh=4.0, showMatches=None) -> Union[None, np.array, Tuple[np.array, np.array]]:
    # local invariant descriptors from them
    (imageB, imageA) = images
    (kpsA, featuresA) = detectAndDescribe(imageA)
    (kpsB, featuresB) = detectAndDescribe(imageB)
    # match features between the two images
    M = matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
    # if the match is None, then there aren’t enough matched
    if M is None:
        return None

    # otherwise, apply a perspective warp to stitch the images together
    (matches, H, status) = M
    result = cv2.warpPerspective(
        imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

    # check to see if the keypoint matches should be visualized
    if showMatches is not None:
        vis = drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
        # return a tuple of the stitched image and the visualization
        cv2.imshow(showMatches, vis)

    # return the stitched image
    return result


def detectAndDescribe(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect keypoints in the image
    orb = cv2.ORB_create()

    kps, features = orb.detectAndCompute(image, None)
    # convert the keypoints from KeyPoint objects to NumPy arrays
    kps = np.float32([kp.pt for kp in kps])
    # return a tuple of keypoints and features
    return (kps, features)


def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
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


def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
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


def main():
    # We can use the ArgParser module to make it super fancy
    DATASET = 'mountains'

    dataset = []
    if DATASET == 'sign':
        dataset = [cv2.imread(f"panos/Sign/sign{idx}.jpg") for idx in range(3)]
    elif DATASET == 'pool':
        dataset = [cv2.imread(f"panos/Pool/pool{idx}.jpg") for idx in range(3)]
    elif DATASET == 'mountains':
        dataset = [cv2.imread(f"panos/Mountains/mountains{idx}.jpg") for idx in range(5)]
    else:
        print("DATASET must be 'sign' or 'pool' or 'mountains'")
        sys.exit(0)

    # numImages = len(dataset)

    # result = stitch([dataset[0], dataset[1]])
    # result = stitch([result, dataset[2]])
    # result = stitch([result, dataset[3]])
    # result = stitch([dataset[2], dataset[3]])
    result = dataset[4]
    result = stitch([dataset[3], result])
    result = stitch([dataset[2], result])
    result = stitch([dataset[1], result])
    result = stitch([dataset[0], result])
    stitched = result

    
    # resultLeft = stitch([dataset[0], dataset[1]])#, showMatches="Keypoint matches left")
    # cv2.imshow("Result left", resultLeft)

    # resultRight = stitch([dataset[3], dataset[4]])#, showMatches="Keypoint matches right")
    # cv2.imshow("Result right", resultRight)

    # resultRightCenter = stitch([dataset[2], resultRight])#, showMatches="Keypoint matches right-center")
    # cv2.imshow("Result right-center", resultRightCenter)

    # result = stitch([resultLeft, resultRightCenter], showMatches="Keypoint matches left-center")
    # cv2.imshow("Result", result)

    # stitched = result

    # ((1  2)   3   (4   5))
    #  len 6
    # mid 3

    # stitch the images together to create a panorama
    # left, mid, right, *_ = dataset
    # result = stitch([mid, right], showMatches=True)
    # if result is None:
    #     print("Failed to stitch first two")
    #     sys.exit(0)

    # stitched, vis = result

    # result = stitch([left, stitched], showMatches=True)
    # if result is None:
    #     print("Failed to stitch last two")
    #     sys.exit(0)
    # stitched, vis = result

    # cv2.imshow("Keypoint Matches", vis)
    cv2.imshow("Result", stitched)
    cv2.waitKey(0)
    # cv2.imwrite("output.jpg", stitched)


if __name__ == "__main__":
    main()
