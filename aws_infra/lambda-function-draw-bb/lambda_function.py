import json
import cv2
import boto3
import numpy as np

def lambda_handler(event, context):
    print('dw_event', event)
    if len(event) < 1:
        print("dw_ empty event received, returning to flush stream")
        return
    
    bucket = event['bucket']
    key = event['key']
    s3 = boto3.client('s3')
    img_raw = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
    #print(img_raw)
    nparr = np.fromstring(img_raw, np.uint8)
    imgData = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    print("Starting add of bboxes")
    # iterate over objects and draw bounding boxes   
    i = 0
    for item in event['items']:
        i+=1
        print("Bbox #", i)
        item_class = str(item['class'])
        confidence = item['confidence']
        left = item['left']
        top = item['top']
        width = item['width']
        height = item['height']
        right = left + width
        bottom = top + height
        label = item_class + "(" + str(confidence) + ")"
        # Draw bounding box
        imgHeight, imgWidth, _ = imgData.shape
        thick = (imgHeight + imgWidth) // 900 * 2
        color = (0,0,255)
        cv2.rectangle(imgData, (left, top), (right, bottom), color, thick)
        cv2.putText(imgData, label, (left, top - 16), 0, 1e-3 * imgHeight, color, 2)
        
    #cv2.imwrite(tmp_image_filename, imgData)
    image_bytes = cv2.imencode('.jpg', imgData)[1].tobytes()
    s3.put_object(Body=image_bytes, Bucket=bucket, Key=key)
        
    print("Finished adding bboxes")
            
    return {
        'statusCode': 200
    }
