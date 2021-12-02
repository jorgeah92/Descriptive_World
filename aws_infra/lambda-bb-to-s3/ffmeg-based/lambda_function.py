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

region_name = "us-west-2"
bucket_name = "descriptiveworld-demo-images"
kvs_name = 'descriptiveworld-demo-kvs'
kvs_ARN = 'arn:aws:kinesisvideo:us-west-2:769212126689:stream/descriptiveworld-demo-kvs/1636338792582'

        
def lambda_handler(event, context):
    #print('dw_event', event)
    #print('dw_context', context)
    if False: # should be False for normal operation
        print('dw_manual bypass to clear the stream')
        return
    
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
        print('dw_data', data)
        
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
        response_video = video_client.get_media_for_fragment_list(
            StreamARN=kvs_ARN,
            Fragments=[
                fragmentNumber,
            ]
        )
        print(response_video)
        
        # pull in stream then chunk it
        payload_video = response_video["Payload"]
        #print(payload_video)

        # write chunk to temp directory
        tmp_chunk_filename = "/tmp/" + fragmentNumber + ".mkv"
        with open(tmp_chunk_filename, 'wb') as video_file:
            video_chunk = payload_video.read()
            print("Chunk read")
            video_file.write(video_chunk)
            print("chunk written")
        #f.close()
        
        # extract frame with ffmpeg
        tmp_image_filename = "/tmp/" + fragmentNumber + ".jpg"
        print("Trying conversion from ", tmp_chunk_filename, " to ", tmp_image_filename)
        os.system("/opt/bin/ffmpeg -y -i {0} -ss 00:00:00 -vframes:v 1 {1}".format(tmp_chunk_filename, tmp_image_filename))
        print("Extracted image from chunk")
        
        # Store image file in S3 (when stored in variable)
        #bucket.put_object(Key=fragment_filename, Body=img)
        # storage image in S3 from /tmp
        ##s3.Object(bucket_name, fragment_filename).put(Body=open(tmp_image_filename, 'rb'))
        # make a copy as current
        ##s3.Object(bucket_name, current_filename).copy_from(CopySource=bucket_name + "/" + fragment_filename)
        s3.Object(bucket_name, current_filename).put(Body=open(tmp_image_filename, 'rb'))
        print("Copied image to S3 bucket")
        
        # ADD BOUNDING BOX CODE HERE, REPLACE CURRENT IMAGE WITH BOUNDING BOX IMAGE   
        items = json.loads(json.dumps(data['objects']))
        #print(items)

        # call our other lambda with CV2 installed
        print("Calling bounding box draw lambda")
        lambda_client = boto3.client('lambda')
        lambda_response = lambda_client.invoke(FunctionName="descriptiveworld-demo-lambda-function-draw-bb",
                                           InvocationType='RequestResponse',
                                           Payload=json.dumps({
                                                    "bucket": bucket_name,
                                                    "key": current_filename,
                                                    "items": items
                                                })
                                           )
        print(lambda_response)
        print("Returned from bbox lambda")
        
        print("END OF EXECUTION")
          
        return 'Successfully processed {} records.'.format(len(event['Records']))