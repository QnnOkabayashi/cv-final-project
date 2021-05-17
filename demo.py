import cv2
import numpy as np

WINDOW_NAME = "Panorama Stitcher"
WINDOW_X = 25
WINDOW_Y = 25

TARGET_DISPLAY_SIZE_WIDTH = 400
TARGET_DISPLAY_SIZE_HEIGHT = 400

def show_text_screen(text):
    menu = np.full((TARGET_DISPLAY_SIZE_HEIGHT, TARGET_DISPLAY_SIZE_WIDTH, 1),
            0, np.uint8)

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

    while True:
        k = cv2.waitKey(5)
        if k >= 0:
            break

    return k


def display_img_on_screen(image):
    while True:
        cv2.imshow(WINDOW_NAME, image)
        k = cv2.waitKey(5)
        if k >= 0:
            break
        

def display_menu():
    menuText = [
            'Panorama Stitching Menu',
            '',
            'Datasets',
            '[1] - Mountains',
            '[2] - Sign',
            '[3] - Pool',
            '',
            'ESC - QUIT'
            ]

    while True:
        return show_text_screen(menuText)


def display_dataset_menu(DATASET):
    menuText = [
            f'{DATASET.title()} Menu',
            '',
            'Actions',
            '[1] - Scroll through Dataset Images',
            '[2] - Display Stitching Result',
            '[3] - Pool',
            '',
            'ESC - Go Back to Main Menu'
            ]

    while True:
        return show_text_screen(menuText)


def dataset_demo(images):
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
        
        k = cv2.waitKey(5)

        if k == ord(' ') or k == ord(']'):
            idx = (idx + 1) % n
        elif k == ord('['):
            idx = (idx + n - 1) % n
        elif k == 27:
            return


def main():
    wflags = cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_AUTOSIZE

    cv2.namedWindow(WINDOW_NAME, wflags)
    cv2.moveWindow(WINDOW_NAME, WINDOW_Y, WINDOW_X)

    DATASET = None
    images = []
    
    image1 = np.zeros((250, 250, 3), dtype=np.uint8)
    image1[True] = (255,0,0)
    
    image2 = np.zeros((250, 250, 3), dtype=np.uint8)
    image2[True] = (255,255,0)

    images.append(image1)
    images.append(image2)

    result = np.zeros((250,250,3), dtype=np.uint8)
    result[True] = (0,0, 255)

    key = display_menu()
    if key == 27:
        return
    elif key == ord("1"):
        DATASET = "car5" 
        dataset_demo(images)
        display_img_on_screen(result)
    elif key == ord("2"):
        DATASET = "car6"
    elif key == ord("3"):
        DATASET = "HEY"

main()