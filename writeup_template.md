## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"
[image1]: ./output_images/undistort-false.jpg "Before"
[image1.1]: ./output_images/undistort-true.jpg "Undistorted"
[image2]: ./output_images/binary_combo.jpg "Binary Example"
[image3]: ./output_images/warped-binary.jpg "Warp Example"
[image4]: ./output_images/tracked1-part-4-metrics.jpg "Fit Visual"
[image5]: ./output_images/tracked2-part-5-centerline.jpg "Output"
[video1]: ./project_video.mp4 "Video"
[video2]: ./output_tracked.mp4 "Video output"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 7 through 46 of the file called `calibrate.py`).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]
![alt text][image1.1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # 108-112 through # in `image_gen.py`). Here's an example of my output for this step.

![alt text][image2]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `cv2.warpPerspective`, which appears in lines 126 through 129 in the file `image_gen.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook). The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points. I chose the hardcode the source and destination points in the following manner:
Warped + binary:

![alt text][image3]

```python
    src = np.float32([[img.shape[1]*(.5-mid_width/2), img.shape[0]*height_pct], [img.shape[1]*(.5+mid_width/2), img.shape[0]*height_pct],
                      [img.shape[1]*(.5+bot_width/2), img.shape[0]*bottom_trim], [img.shape[1]*(.5-bot_width/2), img.shape[0]*bottom_trim]])
    dst = np.float32([[offset, 0], [img_size[0]-offset, 0],
                      [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])
```

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I fit the polynomial using the numpy method polyfit, defining the corners in the next code:

```python
    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_fitx = left_fit[0]*yvals*yvals + \
        left_fit[1]*yvals + left_fit[2]
    left_fitx = np.array(left_fitx, np.int32)

    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = right_fit[0]*yvals*yvals + \
        right_fit[1]*yvals + right_fit[2]
    right_fitx = np.array(right_fitx, np.int32)
```

![alt text][image4]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 216 # through # 217 in my code in `image_gen.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 203 through 209 in my code in `image_gen.py`. Here is an example of my result on a test image:

![alt text][image5]

---

### Pipeline (video)

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_tracked.mp4)

---

### Discussion

Project still need more polyshing because, lines are not perfect at the top right and left of polynomial, especially when they shake in the turns. Probably the height of the detection line can be a litle bit smaller, that will help reduce the noises in line detection. Also the algorithm should be improved using with different light conditions, so we need to add more colors filter when the image have high contrast or make the logic to apply different filters according to the light conditions(day, night, rain, fog)
