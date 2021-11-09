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
import cv2

import numpy as np
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

def input_fn(input_data_url, content_type):
    # deserializes the prediction input
    print("bcj_start input_fn")
    print("bcj_ input_data type:", type(input_data_url))

    if type(input_data_url) == 'str':
        r = requests.get(input_data_url)
        img_str = r.content

    else:
        img_str = input_data_url
        
    img_bstr = np.fromstring(bytes(img_str), np.uint8)
    im0s = cv2.imdecode(img_bstr, cv2.IMREAD_COLOR) #BGR?

    if im0s is None:
        print('bcj_ no image data found at location:', input_data_url)
        print('bcj_ request response:', r)
        return None
    
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
    return [img, im0s]  # returning im0s per yolov5 code for output_fn (gn calc)

def predict_fn(input_data, model):
    # calls the model on the deserialized data.
    print("start predict_fn")
    
    if (input_data is None):
        print('bcj_ input_data is None')
        return None
    if (model is None):
        print('bcj_ model is None')
        return None
    
    img = input_data[0]
    im0s = input_data[1] # pass-through for output_fn (gn calc)

    pred = model(img)[0]
    pred = non_max_suppression(pred, opt['conf_thres'], opt['iou_thres'], opt['classes'], opt['agnostic_nms'], max_det=opt['max_det'])
    
    print("bcj_end predict_fn")
    return [pred, im0s]  # returning im0s per yolov5 code for output_fn (gn calc)

def output_fn(prediction_output, accept):
    # serializes the prediction output.
    print("bcj_start output_fn")
    
    if prediction_output is None:
        print('bcj_ prediction_output is None')
        return None

    pred = prediction_output[0] 
    im0s = prediction_output[1]
    
    # Process predictions
    out_results = []
    for i, det in enumerate(pred):  # per image
        if len(det):
            s = ''
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {opt['names'][int(c)]}{'s' * (n > 1)}, "  # add to string
            print('Detected', s)

            for *xyxy, conf, cls in reversed(det):
                gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                result = {
                    'class_num': int(cls), 
                    'class_name': opt['names'][int(cls)], 
                    'bbox': xywh, 
                    'conf': float(conf)
                }
                print(result)
                out_results.append(result)

    print("bcj_end output_fn")
    return out_results

if False:#__name__ == "__main__": # remove False to run from CLI
    # test harness code: run this from CLI before deploying to Sagemaker Endpoint
    print("bcj_start main")
    
    tst_model = model_fn('best.pt')
    
    #input_data_url = './data/images/bus.jpg'
    input_data_url = 'https://media.gq.com/photos/60f9c697101cc04fad71e5cf/master/pass/BEST-BASICS-1.jpg'
    #input_data_url = 'https://descriptiveworld-datasets.s3.us-west-2.amazonaws.com/Fashion_Product_Images/images/10003.jpg'
    r = requests.get(input_data_url)
    img_bstr = r.content
    tst_input = input_fn(img_bstr, 'url')
    
    tst_pred = predict_fn(tst_input, tst_model)
    
    tst_out = output_fn(tst_pred, 'accept')
    print(tst_out)
    print("bcj_end main")
