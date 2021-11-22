# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Adapted to run as SageMaker Endpoint by Blair Jones on 03 Nov 2021.
Inspired by example at:  https://sagemaker-examples.readthedocs.io/en/latest/aws_sagemaker_studio/frameworks/pytorch_cnn_cifar10/pytorch_cnn_cifar10.html

"""

print('start of bcj_sm_ep_detect.py')

import os
import sys
from pathlib import Path
import requests
import json
import cv2
import datetime
import numpy as np
import base64
import torch
import torch.backends.cudnn as cudnn

from utils.augmentations import letterbox

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, strip_optimizer, xyxy2xywh, LOGGER
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync

### Global variable
opt = {}
opt['pt']=True  # always assume pytorch
opt['device']=''  # # cuda device, i.e. 0 or 0,1,2,3 or cpu or ''=pytorch chooses
opt['imgsz']=640  # inference size (pixels)
opt['conf_thres']=0.25  # confidence threshold
opt['iou_thres']=0.45  # NMS IOU threshold
opt['max_det']=1000  # maximum detections per image
opt['stride']=64
opt['device']=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
opt['classify']=False # run 2nd stage classifier
opt['agnostic_nms']=False  # class-agnostic NMS
opt['half']=False  # use FP16 half-precision inference
opt['classes']=None  # filter by class: --class 0, or --class 0 2 3
opt['names'] = [f'class{i}' for i in range(1000)]  # assign defaults

def model_fn(model_path):
    # loads the model from disk. This function must be implemented.
    print("bcj_start model_fn")
    print('bcj_ model_path', model_path)
    
    if os.path.exists(model_path):
        path_conts = os.listdir(model_path)
        if len(path_conts) < 1:
            print('bcj_ no model files')
        else:
            print('bcj_ model_path contents\n', path_conts)
    else:
        print('bcj_ model path not found on instance')
    
    # Initialize
    opt['cur_device'] = select_device(opt['device'])
    opt['half'] &= opt['cur_device'].type != 'cpu'  # half precision only supported on CUDA

    # Load model
    print("bcj_start load model")
    if opt['pt']:
        #reference:  https://stackoverflow.com/questions/68150444/aws-sagemaker-fails-loading-pytorch-pth-weights
        model = attempt_load(os.path.join(model_path, 'model.pt'), map_location=opt['cur_device'])
        opt['stride'] = int(model.stride.max())  # model stride
        opt['names'] = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if opt['half']:
            model.half()  # to FP16
            
    print("bcj_start check image size")
    imgsz = check_img_size(opt['imgsz'], s=opt['stride'])  # check image size
    
    print("bcj_start run once")
    if opt['pt'] and opt['cur_device'].type != 'cpu':
        model(torch.zeros(1, 3, *opt['imgsz']).to(opt['cur_device']).type_as(next(model.parameters())))  # run once
        
    print("bcj_end model_fn")
    return model

def input_fn(input_data, content_type='JPEG', *argparams, **kwparams):#custom_attributes=None):
    # Expects input_data in format bytestring of an image (ex. JPEG) or URL (string of http://)
    # deserializes the prediction input
    print("bcj_start input_fn")
    print("bcj_ input_data type:", type(input_data))
    print("bcj_ argparams", argparams)
    print("bcj_ kwparams", kwparams)


    if content_type == 'URL': # this is used for debugging when invoking from CLI
                                #  DO NOT USE for endpoint deployment
        r = requests.get(input_data)
        img_str = r.content
    else: # this is the path normally used for endpoint deployment
        img_str = input_data
        
    img_bstr = np.fromstring(bytes(img_str), np.uint8)
    im0s = cv2.imdecode(img_bstr, cv2.IMREAD_COLOR) #BGR?

    if im0s is None: # should only ever occur when calling from CLI in debugging mode (not endpoint deployed)
        print('bcj_ no image data found at location:', input_data)
        if r is not None:
            print('bcj_ request response:', r)
        err = createOutputTemplate()
        err['status'] = 'Error'
        err['status-description'] = 'Problem with image data'
        return err
    
    # Padded resize
    img = letterbox(im0s, opt['imgsz'], stride=opt['stride'], auto=opt['pt'])[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    
    img = torch.from_numpy(img).to(opt['cur_device'])
    img = img.half() if opt['half'] else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
          
    print("bcj_end input_fn")
    return [img, im0s]  # returning im0s per yolov5 code for output_fn ("gn" calc)

def predict_fn(input_data, model):
    # calls the model on the deserialized data.
    print("start predict_fn")
    
    err = createOutputTemplate()
    err['status'] = 'Error'

    if (input_data is None):
        print('bcj_ input_data is None')
        err['status-description'] = 'No input data found'
        return err
    if (model is None):
        print('bcj_ model is None')
        err['status-description'] = 'No model found'
        return err
    
    img = input_data[0]
    im0s = input_data[1] # pass-through for output_fn ("gn" calc)

    pred = model(img)[0]
    pred = non_max_suppression(pred, opt['conf_thres'], opt['iou_thres'], opt['classes'], opt['agnostic_nms'], max_det=opt['max_det'])
    
    print("bcj_end predict_fn")
    return [pred, img, im0s]  # returning img and im0s per yolov5 code for output_fn (gn calc)

def output_fn(prediction_output, accept):
    # serializes the prediction output.
    print("bcj_start output_fn")
    
    if prediction_output is None:
        print('bcj_ prediction_output is None')
        err = createOutputTemplate()
        err['status'] = 'Error'
        err['status-description'] = 'Prediction output was None'
        return err

    pred = prediction_output[0] 
    img  = prediction_output[1]
    im0s = prediction_output[2]
    
    img_shape = img.shape
    im0s_shape = im0s.shape
    
    # Process predictions
    output = createOutputTemplate()
    output["bounding-box-attribute-name"]["image_size"][0]["width"] = im0s_shape[1]
    output["bounding-box-attribute-name"]["image_size"][0]["height"] = im0s_shape[0]
    output["bounding-box-attribute-name"]["image_size"][0]["depth"] = im0s_shape[2]
    
    detections = False
    
    for i, det in enumerate(pred):  # per image
        if len(det):
            detections = True
            
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img_shape[2:], det[:, :4], im0s_shape).round()
            
            s = ''
            tot_dets = 0 # total detections
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                tot_dets += n
                s += f"{n} {opt['names'][int(c)]}{'s' * (n > 1)}, "  # add to string
            print('Detected', s)
            output["num-detected-objects"] = int(tot_dets)

            for *xyxy, conf, cls in reversed(det):
                gn = torch.tensor(im0s_shape)[[1, 0, 1, 0]]  # normalization gain whwh
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                output["bounding-box-attribute-name"]["annotations"].append(createAnnotation(int(cls), im0s_shape, *xywh))
                output["bounding-box-attribute-name-metadata"]["objects"].append(createMetadataObjects(round(float(conf),2)))
                output["bounding-box-attribute-name-metadata"]["class-map"][str(int(cls))] = opt['names'][int(cls)]

    print("bcj_output", output)
    
    output["original-image"] = base64.b64encode(im0s).decode() # add after logging to cloudwatch

    print("bcj_end output_fn")
    return json.dumps(output)

def createNormAnnotation(cls_id, ctr_x_pct, ctr_y_pct, w_pct, h_pct):
    # NOTE:  Bbox values expressed as percentage of original image width and height
    return {"class_id": cls_id, "center_x_pct": round(ctr_x_pct, 4), "center_y_pct": round(ctr_y_pct, 4),
                        "width_pct": round(w_pct,4), "height_pct": round(h_pct,4)}

def createAnnotation(cls_id, img_dim, ctr_x_pct, ctr_y_pct, w_pct, h_pct):
    # Version 1
    img_height = float(img_dim[0])
    img_width = float(img_dim[1])
    width = int(img_width*w_pct)
    height = int(img_height*h_pct)
    left = int(ctr_x_pct*img_width - width/2.)
    top = int(ctr_y_pct*img_height - height/2.)
    return {"class_id": cls_id, "left": left, "top": top,
                        "width": width, "height": height}

    # Version 2:  not used
    # NOTE:  Bbox values expressed as percentage of original image width and height
    return {"class_id": cls_id, "center_x_pct": round(ctr_x_pct, 4), "center_y_pct": round(ctr_y_pct, 4),
                        "width_pct": round(w_pct,4), "height_pct": round(h_pct,4)}


def createMetadataObjects(conf):
    return {"confidence": conf}
            
def createOutputTemplate():
    template = {
        "source-ref": "TBD",
        "num-detected-objects": 0,
        "bounding-box-attribute-name":
        {
            "image_size": [{ "width": 0, "height": 0, "depth":0}],
            "annotations": [] # add to this by invoking createAnnotation()
        },
        "bounding-box-attribute-name-metadata":
        {
            "objects": [],
            "class-map": {},
            "type": "descriptiveworld/object-detection",
            "human-annotated": "no",
            "creation-date": str(datetime.datetime.now()),
            "job-name": "descriptive_world_identify_garments"
        },
        "original-image": "",
    }
    return template


if False:#__name__ == "__main__": # remove False to run from CLI
    # test harness code: run this from CLI before deploying to Sagemaker Endpoint
    print("bcj_start main")
    
    tst_model = model_fn('./')
    
    #input_data_url = './data/images/bus.jpg'
    input_data_url = 'https://d2ph5fj80uercy.cloudfront.net/04/cat1600.jpg'
    #input_data_url = 'https://descriptiveworld-datasets.s3.us-west-2.amazonaws.com/Fashion_Product_Images/images/10003.jpg'
    r = requests.get(input_data_url)
    img_bstr = r.content
    tst_input = input_fn(img_bstr, 'url')
    
    tst_pred = predict_fn(tst_input, tst_model)
    
    tst_out = output_fn(tst_pred, 'accept')
    print(tst_out)
    print("bcj_end main")

 