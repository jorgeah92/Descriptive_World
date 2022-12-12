# Descriptive World
## Descriptive World Project Outfit

### Authors - Blair Jones, Jorge Hernandez, Jack Wang, Matt White

Project outfit is a capstone project of four students for the Master in Information and Data Science program at UC Berkeley. The idea of the project was to construct an application for clothing type classification directed toward the visually impaired and utlimately an api for building future features for the target audience. In this project a computer vision model is constructed, using YOLOv5, that is able detect 11 types of clothes, color, and texture/pattern of the clothing in both an AWS and IOS implementation.

To read more about the project, check out our site [Descriptive World](https://www.descriptiveworld.com/project-outfits/) for more detailed break down of the project.


### Folders contained in this repo

 **./CV** - Computer Vision Components (object & action detection); This folder contains all of the computer vision models tried throughout the project. This includes a YOLO model, EfficientDet model, Mask R-CNN model, and Faster-RCNN model.<br>
 **./NLP** - Natural Language Processing Components (voice commands)<br>
 **./TTS** - Text-to-Speech (narration)<br>
 **./aws_infra** - Infrastructure Code for AWS (front end, video pipeline, API gateway, lambdas); Includes all of the code used to construct the AWS implementation of this project.<br>
 **./IOS** - IOS application code; Includes all of the code used to build the IOS implementation of this project.<br>
 **./Misc** - Miscellanious files; Includes files like EDA of datasets, extra features, and data preparation.


