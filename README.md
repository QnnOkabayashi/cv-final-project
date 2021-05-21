# Panoramic Stitching
Our program grants the user the ability to import and select a panoramic video. Our program then takes this video and extracts a user-determined number of frames/images from the selected video. Then, our program extracts the keypoints and key features of each image in order to construct a homography for each image. Using *perspective transformations* and *homography stitching*, our program stitches all of the images together to create a single panoramic image.

## Usage
```
$ python main.py
``` 
```
$ python downscale_video.py VIDEOPATH VIDEONAME
```
## Navigating UI
Listed below are the keys our UI currently responds to:

* `ESC` - QUIT/ Go Back to Previous Menu
* `ENTER` - Confirm Selection
* `]` - Next Selection
* `[` - Previous Selection

