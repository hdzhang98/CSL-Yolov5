# CSL-Yolov5
This document describes the configurations and operation steps of the CLS-YOLOv5 on Cornell and Jacquard datasets. Similar to the configuration of the YOLOv5 in its original download address: “https://github.com/ultralytics/yolov5”, the CLS-YOLOv5 address (https://github.com/hdzhang98/CSL-Yolov5) have four subdirectories: 

“/[DataSet](https://github.com/hdzhang98/CSL-Yolov5/tree/master/DataSet/cornell/images)”

“/[models](https://github.com/hdzhang98/CSL-Yolov5/tree/master/models)”

“/[utils](https://github.com/hdzhang98/CSL-Yolov5/tree/master/utils)”

“/[weights](https://github.com/hdzhang98/CSL-Yolov5/tree/master/weights)”

In the CSL-YOLOv5, the images are stored in the subdirectory of “/[DataSet](https://github.com/hdzhang98/CSL-Yolov5/tree/master/DataSet/cornell/images) ”, the codes are put in “/[models](https://github.com/hdzhang98/CSL-Yolov5/tree/master/models)”, the auxiliary tools are put in “/[utils](https://github.com/hdzhang98/CSL-Yolov5/tree/master/utils)”, and the trained network weights for Cornell and Jacquard are put in “/[weights](https://github.com/hdzhang98/CSL-Yolov5/tree/master/weights)”. 

 

The recommended development tools for CSL-YOLOv5 are python==3.8.12 and pytorch==1.6.0. You are suggested

 to configure the development environment using a command “pip install -r requirements.txt”. With this, the CSL-YOLOv5 is ready.

 

Before validating the CSL-YOLOv5, please change the following parameters of “detect.py” to configure the test: 

The parameter “--source” in detect.py (for example: we use “DataSet/jacquard/images” in detect.py) indicates the image directory;

The parameter “--output” in detect.py (for example: we use “DataSet/jacquard/detection” in detect.py) indicates the detection results directory;

The parameter “--weights” in detect.py (for example: we use “./weights/jacquard.pt” in detect.py) indicates the network weights directory.

 

After changing these file directories to any directories you want, then detect images using the command: 

“python detect.py”

 

After the running of the CSL-YOLOv5, the detection result images will be listed in the '--output' path, and a “txt” folder will be added automatically in the '--output' path. The files in the “txt” folder indicate the detailed prediction information with the format: 

“Image_name, confidence, x1, y1, x2, y2, x3, y3, x4, y4, class”. 

 

In “Image_name, confidence, x1, y1, x2, y2, x3, y3, x4, y4, class”, the “Image_name” is the image name of the detection image, “x1, y1, x2, y2, x3, y3, x4, y4” are the x, y coordinates of the four corner points of the detection grasp anchor, the “confidence” is the confidence of the anchor, and “class” is the object name of the target presented in this image.

 

You are welcome to test and validate the CSL-YOLOv5 models. If there are any questions, please contact [hdzhang98@gmail.com](mailto:hdzhang98@gmail.com) and [mhyang@nlpr.ia.ac.cn](mailto:mhyang@nlpr.ia.ac.cn). Thanks!