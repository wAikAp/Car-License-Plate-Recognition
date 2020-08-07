# Car-License-Plate-Recognition
Car License Plate numbers and charactes Recognition by TensorFlow and Pytesseract

- Recognize the numbers and characters of license plate for Hong Kong.
- Only support Hong Kong license plate.
- Suggested image size >= 1024 x 800

## Environment and Software package
- Python 3.6.8
- [Python virtual environments (venv)](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/ "Python virtual environments (venv)")
- [TensorFlow 1.14](https://www.tensorflow.org/install/pip "- TensorFlow 1.14")
- [Tensorflow Model 1.13 and Object detection API 1.13](https://github.com/tensorflow/models/tree/r1.13.0 "Tensorflow Model and Object detection")
- [OpenCV](https://github.com/skvark/opencv-python "OpenCV")
- [pytesseract](https://github.com/madmaze/pytesseract "pytesseract")
- [matplotlib](https://matplotlib.org/ "matplotlib") 
- [Labelimg](https://github.com/tzutalin/labelImg "Labelimg")

## Project idea
1. Use TensorFlow object detection to recognize the position of the license plate on the image.
2. dependence the position to do the OCR process by using the by tesseract to recognize those numbers and characters.




# Implementation
## License Plate Detector
Collect the car license plate data set of more than 100 images. I found the Hong Kong license plate from [car-plate-data-set](https://www.flickr.com/photos/jw713713/albums/with/72157641638373464 "car-plate-data-set") which is free and open source for usage.
[![](https://raw.githubusercontent.com/wAikAp/Car-License-Plate-Recognition/master/readmeImg/cap.png)](https://raw.githubusercontent.com/wAikAp/Car-License-Plate-Recognition/master/readmeImg/cap.png)
Then use the labelimg to label and locate the license plate, and save as xml for prepare the training process, so you need to prepare 2 sets of data(one training one testing) which scale as 80%(train) and 20%(test).

Next, download the [TensorFlow Model ](https://github.com/tensorflow/models "TensorFlow Model/tree/r1.13.0")to your machine, then setup and build the Object detection API. When all the things are setup completed and successfully, then next can start the training process.
More about upper step (how to create your own object detector using the TensorFlow object detection API) you can reference here: https://towardsdatascience.com/creating-your-own-object-detector-ad69dda69c85 (as same as me reference here).
**Note:** My Case Using Tensorflow CPU And Faster_rcnn_inception_v2_coco_2018_01_28 For Training.

## Number Characters Recognizer
In the Python programming([source code are here](https://github.com/wAikAp/Car-License-Plate-Recognition/blob/master/Car-License-Plate-Recognition.ipynb "source code are here")), use the TensorFlow detection program to find the license plate detected position(xmin, ymin, xmax, ymax). 
[![](https://raw.githubusercontent.com/wAikAp/Car-License-Plate-Recognition/master/output_images/13562150034_aa4c7b050e_o.jpg)](https://raw.githubusercontent.com/wAikAp/Car-License-Plate-Recognition/master/output_images/13562150034_aa4c7b050e_o.jpg)

![](https://raw.githubusercontent.com/wAikAp/Car-License-Plate-Recognition/master/output_images/31299736077_4f274523d6_b.jpg)

**Then crop that position use** `tf.image.crop_to_bounding_box`

`cropped_image = tf.image.crop_to_bounding_box(image_np, int(d_ymin), int(d_xmin), int(d_ymax - d_ymin), int(d_xmax - d_xmin))`

------
Then use openCV set the image from RGB  to gary
`gray = cv2.cvtColor(detect_cropped_image, cv2.COLOR_BGR2GRAY)`
`gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]`
`gray = cv2.medianBlur(gray, 3)`

[![](https://raw.githubusercontent.com/wAikAp/Car-License-Plate-Recognition/master/readmeImg/gary.png)](https://raw.githubusercontent.com/wAikAp/Car-License-Plate-Recognition/master/readmeImg/gary.png)

![](https://raw.githubusercontent.com/wAikAp/Car-License-Plate-Recognition/master/readmeImg/gary2.png)
------

Lastly use the pytesseract to do the OCR.
Set the config that means here is the pytesseract OCR white list for each characters.
`custom_config = r'-c tessedit_char_whitelist=ABCDEFGHJKLMNPQRSTUVWXYZ1234567890 --psm 6'`

OCR process and recognize the numbers and characters
`plate_num = pytesseract.image_to_string(gray,config=custom_config)`

------

### Finally the result looks like:
![](https://raw.githubusercontent.com/wAikAp/Car-License-Plate-Recognition/master/readmeImg/result.png)
![](https://raw.githubusercontent.com/wAikAp/Car-License-Plate-Recognition/master/readmeImg/result2.png)

------
# End
This project just a simple idea for how to recognize the car license plate it is not > 50% accurate, and nows day have a lot of machine learning platforms and new tech, so here just an example for combining the technique, don't let the frame and rule fixed your ideas! hope this project can help you get some new ideas. 




