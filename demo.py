##############################################################################
#
# File: demo.py
# Authors: Quinn Okabayashi, Theron Mansilla
# Course: ENGR 27
# Date: May 18 2021
#
# Resources:
# * UI Inspiration:
# * https://github.com/swatbotics/mnist_pca_knn/blob/main/mnist_pca_knn.py
#
##############################################################################
#
# This file provides UI utilities.
#
##############################################################################

import cv2
import numpy as np
import sys
import os
from typing import Tuple, List, Union
from video_slicer import get_key_frames

WINDOW_NAME = "Panorama Stitcher"
WINDOW_X = 25
WINDOW_Y = 25

TARGET_DISPLAY_SIZE_WIDTH = 400
TARGET_DISPLAY_SIZE_HEIGHT = 400


def show_text_screen(text: List[str]) -> None:
    menu = np.zeros(shape=(TARGET_DISPLAY_SIZE_HEIGHT, TARGET_DISPLAY_SIZE_WIDTH, 1),
                    dtype='uint8')

    x = 10
    y = 20
    l = 20
    sz = 0.5

    for line in text:
        if line:
            cv2.putText(menu, line, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        sz, (255, 255, 255), 1, cv2.LINE_AA)

        y += l

    cv2.imshow(WINDOW_NAME, menu)


def display_dataset_menu(dataset: str) -> None:
    menu_text = [
        f'{dataset} menu',
        '',
        'actions',
        '[1] - scroll through dataset images',
        '[2] - display stitching result',
        '[3] - save stitched result',
        '',
        'esc - go back to main menu'
    ]

    show_text_screen(menu_text)


def get_user_choice(header: str, options: List[str], idx: int = 0) -> Union[None, int]:
    menu_text = [
        header,
        '',
        '',
        '',
        'Hit [ or ] for prev/next',
        'Hit Enter to select',
    ]
    n = len(options)

    while True:
        menu_text[2] = f'[{idx + 1}/{n}] - {options[idx]}'
        show_text_screen(menu_text)

        k = cv2.waitKey(0)

        if k in (ord(' '), ord(']')):
            idx = (idx + 1) % n
        elif k == ord('['):
            idx = (idx + n - 1) % n
        elif k == 27:
            return None
        elif k == 13:
            return idx


def select_video_menu() -> Union[None, Tuple[np.array, str]]:
    while True:
        # Select a video
        video_names = [name for name in os.listdir(
            'videos') if name.endswith('.mp4')]
        idx = get_user_choice("Video Selection Menu", video_names)
        if idx is None:
            return None

        dataset_filename = video_names[idx]
        dataset_name = dataset_filename[:-4]
        filename = os.path.join("videos", f"{dataset_filename}")

        while True:
            # Select a frame count
            frame_choices = list(map(str, np.arange(1, 11)))
            idx = get_user_choice("Frame Count Menu", frame_choices, 2)
            if idx is None:
                break

            frame_count = int(frame_choices[idx])

            images = get_key_frames(
                filename, target=0.2, count=frame_count)
            if images is None:
                return None

            return images, dataset_name


def dataset_demo(images: np.array) -> None:
    instructions = [
        'Dataset Images',
        '',
        'Hit [ or ] for prev/next image',
        'Hit ESC when done',
        '',
        'Hit any key to begin',
    ]

    show_text_screen(instructions)

    idx = 0
    n = len(images)

    while True:
        cv2.imshow(WINDOW_NAME, images[idx])

        k = cv2.waitKey(0)

        if k == ord(' ') or k == ord(']'):
            idx = (idx + 1) % n
        elif k == ord('['):
            idx = (idx + n - 1) % n
        elif k == 27:
            return
