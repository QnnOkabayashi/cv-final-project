import cv2
import numpy as np
import matplotlib.pyplot as plt


def bf_orb(img1: np.array, img2: np.array, num_matches=10, show_matches=False) -> int:
    orb: cv2.ORB = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf: cv2.BFMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches: List[cv2.DMatch] = bf.match(des1, des2)

    matches: List[cv2.DMatch] = sorted(matches, key=lambda x: x.distance)[:num_matches]

    if show_matches:
        img_matches: np.array = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.imshow(img_matches), plt.show()

    return matches


if __name__ == '__main__':
    def main():
        img1 = cv2.imread("project4/data2/pool_left.jpg")
        img2 = cv2.imread("project4/data2/pool_right.jpg")

        bf_orb(img1, img2, show_matches=True)
    main()
