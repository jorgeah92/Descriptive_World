# Descriptive World
## Descriptive World Project Outfit

### Authors - Blair Jones, Jorge Hernandez, Jack Wang, Matt White

Project outfit is a capstone project of four students for the Master in Information and Data Science program at UC Berkeley. The idea of the project was to construct an application for clothing type classification directed toward the visually impaired and utlimately an api for building future features for the target audience. We built an application for both the IOS platform and web interface using AWS components. The application used a computer-vision model to identify clothing via a live feed, either webcam or phone camera, then describes the item out-loud back to the individual. The read out description includes details about the item such as clothing type, clothing pattern, and clothing color. For this project, we used the YOLOv5 architecture and trained two models, one for clothing type and the other for pattern recognition. The models are able to identify 11 different pieces of clothing and 7 different patterns that the clothing items display. Our models perform clothing identification with an average mAP@0.5 rating of 0.7. 

The AWS portion of the project utilize SageMaker, Kinesis Video Stream, Kinesis Datastream, S3 Buckets, Polly, API Gateway, Lex, and Lambdas from the AWS suite of applications/APIs.

To read more about the project, check out our site [Descriptive World](https://www.descriptiveworld.com/project-outfits/) for more detailed break down of the project.


### Folders contained in this repo

 **./CV** - Computer Vision Components (object & action detection); This folder contains all of the computer vision models tried throughout the project. This includes a YOLO model, EfficientDet model, Mask R-CNN model, and Faster-RCNN model.<br>
 **./NLP** - Natural Language Processing Components (voice commands)<br>
 **./TTS** - Text-to-Speech (narration)<br>
 **./aws_infra** - Infrastructure Code for AWS (front end, video pipeline, API gateway, lambdas); Includes all of the code used to construct the AWS implementation of this project.<br>
 **./IOS** - IOS application code; Includes all of the code used to build the IOS implementation of this project.<br>
 **./Misc** - Miscellanious files; Includes files like EDA of datasets, extra features, and data preparation.


