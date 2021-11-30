from __future__ import print_function
import base64
import json
import os
from botocore.exceptions import ClientError
from datetime import datetime
import time
import calendar
import boto3
import codecs
import cv2
import imageio
import pika

region_name = "us-west-2"
bucket_name = "descriptiveworld-demo-images"
kvs_name = 'descriptiveworld-demo-kvs'
kvs_ARN = 'arn:aws:kinesisvideo:us-west-2:769212126689:stream/descriptiveworld-demo-kvs/1636338792582'

# transcode image to jpeg
def transcode_frame(frame):  
    # Encode frame into string for job submission
    #img_str = cv2.imencode('.jpg', frame)[1].tostring()
    #img = cv2.imencode('.jpg', frame)[1]
    print("One frame transocded")
    return img
    
# function to extract frame from a chunk
def get_frame(chunk):
    try:
        fragment = imageio.get_reader(io.BytesIO(chunk), 'ffmpeg')
        for num , im in enumerate(fragment):
            print(num)
            if num % 30 == 0:
                sts = 0
                print("Frame captured")
                break
        print("Returning result")
        return im, sts
        # print(f'Finish one chunk took: {timeit.default_timer() - start_time}')
    except OSError as e:
        print("Broken fragment received")
        sts = 1
        return None, sts
        
def lambda_handler(event, context):
    for record in event['Records']:
        # debug
        #print("Received event: " + json.dumps(event, indent=2))
        payload = base64.b64decode(record['kinesis']['data'])
        #Get Json format of Kinesis Data Stream Output
        result = json.loads(payload)
        print("START OF EXECUTION")
        # debug
        print("Raw Results:", result)
        
        #Get data and load as json
        data = json.loads(json.dumps(result['data']))
        
        # s3 bucket assignments
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        
        # filename settings
        current_filename = "current.jpg"
        fragmentNumber = str(data['fragment-number'])
        fragment_filename = fragmentNumber + ".jpg"

        # retrieve clip from fragment id
        kinesis_client = boto3.client('kinesisvideo', region_name=region_name)
        response = kinesis_client.get_data_endpoint(
            StreamARN=kvs_ARN,
            APIName='GET_MEDIA_FOR_FRAGMENT_LIST'
        )
        print(response)
        
        video_client = boto3.client('kinesis-video-archived-media', endpoint_url=response['DataEndpoint'])
        response = video_client.get_media_for_fragment_list(
            StreamARN=kvs_ARN,
            Fragments=[
                fragmentNumber,
            ]
        )
        print(response)
        
        # pull in stream then chunk it
        stream = response['Payload']
        chunk = stream.read(1024*8*8)
        print("Chunk read")
        # extract frame
        im, sts = get_frame(chunk)
        # transcode image
        img = transcode_frame(im)
         
        
        # ADD BOUNDING BOX CODE HERE, REPLACE CURRENT IMAGE WITH BOUNDING BOX IMAGE   
        
        # Store image file in S3
        bucket.put_object(Key=fragment_filename, Body=img)
        # make a copy as current
        s3.Object(bucket_name, current_filename).copy_from(CopySource=bucket_name + "/" + fragment_filename)
        
        

        
        print("END OF EXECUTION")
          
        return 'Successfully processed {} records.'.format(len(event['Records']))
