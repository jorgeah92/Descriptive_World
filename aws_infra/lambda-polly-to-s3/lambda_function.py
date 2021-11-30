from __future__ import print_function
import base64
import json
import os
from botocore.exceptions import ClientError
from datetime import datetime
import time
import calendar
from boto3 import Session
from boto3 import resource
import codecs

region_name = "us-west-2"
bucket_name = "descriptiveworld-demo-audio"

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
        
        # setup session
        session = Session(region_name=region_name)
        polly = session.client("polly")
        
        # s3 bucket assignments
        s3 = resource('s3')
        bucket = s3.Bucket(bucket_name)
        
        # filename settings
        current_filename = "current.mp3"
        sentence = str(data['sentence'])
        fragmentNumber = str(data['fragment-number'])
        fragment_filename = fragmentNumber + ".mp3"

        # generate speech
        response = polly.synthesize_speech(
        Text=sentence,
        OutputFormat="mp3",
        VoiceId="Salli")
        stream = response["AudioStream"]
        
        # Store files (current and named)
        bucket.put_object(Key=fragment_filename, Body=stream.read())
        # make a copy as current
        s3.Object(bucket_name, current_filename).copy_from(CopySource=bucket_name + "/" + fragment_filename)
        print("END OF EXECUTION")
          
        return 'Successfully processed {} records.'.format(len(event['Records']))
