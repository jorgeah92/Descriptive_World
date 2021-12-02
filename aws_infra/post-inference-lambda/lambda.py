from __future__ import print_function
import base64
import json
import boto3
import os
from botocore.exceptions import ClientError
from datetime import datetime
import time
import calendar
import re

# set threshold for confidence
threshold = 0.75
# set region
region = "us-west-2"
stream_out_polly_name = "descriptive-demo-Post-Inference-Polly"
stream_out_polly_shards = 1
stream_out_imagebb_name = "descriptive-demo-Post-Inference-Image-Bounding-Box"
stream_out_imagebb_shards = 1
# setup kinesis client
kinesis_client = boto3.client('kinesis',region_name=region)

# method to write to the kinesis data stream
def put_to_stream(stream_out_name, data):
    timestamp = calendar.timegm(datetime.utcnow().timetuple())
    
    # JSON formatted payload
    payload = {
        'timestamp': int(timestamp),
        'data': data
    }
    print ("Payload for Kinesis DS: ", payload)
    put_response = kinesis_client.put_record(
        StreamName=stream_out_name,
        Data=json.dumps(payload),
        PartitionKey='inference')
    print("Put Response: ", put_response)
        
# our  entry point            
def lambda_handler(event, context):
    for record in event['Records']:
        payload = base64.b64decode(record['kinesis']['data'])
        
        #Get Json format of Kinesis Data Stream Output
        result = json.loads(payload)
        
        print("START OF EXECUTION")
        # debug
        print("Raw Results:", result)
        #Get FragmentMetaData
        fragment = result['fragmentMetaData']
        print("fragment: " + fragment + "\n")
        #Get FrameMetaData
        frame = result['frameMetaData']
        print("frame: " + frame + "\n")
        #Get StreamName
        streamName = result['streamName']
        print("streamName: " + streamName + "\n")
        #Get SageMaker response and decode
        sageMakerOutput = json.loads(base64.b64decode(result['sageMakerOutput']))
        # Parse JSON into an object with attributes corresponding to dict keys.
        print("sagemaker raw output: " + str(sageMakerOutput))
        # First check to see if this is a new image, if not we don't do any inference
        # TODO
        
        # store the number of detected objects
        numObjects = int(sageMakerOutput['num-detected-objects'])
        
        # IF we have no objects, nothing left to do, let's end this lambda
        if (numObjects == 0):
            print("NO OBJECTS DETECTED, EXITING...")
            print("END OF EXECUTION")
            return None
        
        # record the source (should be full URI to image in S3)
        sourceRef = str(sageMakerOutput['source-ref'])
        originalImage = str(sageMakerOutput['original-image'])
        
        # extract fragment number 
        fragmentNumber = re.search(r'^.*?\bfragmentNumber=(\d+),.*', fragment).group(1)
        
        # although image size can appear with every object identification, we know it will always be the same
        imageSize = dict(sageMakerOutput['bounding-box-attribute-name']['image_size'][0])
        imageSizeHeight = imageSize['height']
        imageSizeWidth = imageSize['width']
        imageSizeDepth = imageSize['depth']
    
        # indices are not unique for foundObjects and boundingBoxConf
        # we will convert them to array of lists
        foundObjects = json.loads(json.dumps(sageMakerOutput['bounding-box-attribute-name']['annotations']))
        boundingBoxConfs = json.loads(json.dumps(sageMakerOutput['bounding-box-attribute-name-metadata']['objects']))
        # retrieve fabric and color predictions
        boundingBoxColors = json.loads(json.dumps(sageMakerOutput['bounding-box-attribute-name-metadata']['color_predictions']))
        boundingBoxFabrics = json.loads(json.dumps(sageMakerOutput['bounding-box-attribute-name-metadata']['fabric_predictions']))
        
        # comes in as a list enclosed in {} let's convert to a dictionary since indices are unique
        boundingBoxClassMap = json.loads(json.dumps(sageMakerOutput['bounding-box-attribute-name-metadata']['class-map']))
    
        # debug
        print ("Number of Objects:", numObjects, "\nImage Size:", imageSize)
        print("Found Objects:", foundObjects, "\nBB Conf:", boundingBoxConfs, "\nBB Class Map:", boundingBoxClassMap, "\n")
        
        polly_sentence = "There are " + str(numObjects) + " items in front of you"
        objects_detected = '['
        # iterate over each one of the object found
        # This is the 'annotations', 'bounding-box-attribute-name-metadata'
        for i in range(0,numObjects):
            foundObject = foundObjects[i]
            # extract confidence level, color and fabric
            boundingBoxConf = boundingBoxConfs[i]['confidence']
            boundingBoxColor = boundingBoxColors[i]['color']
            boundingBoxFabric = boundingBoxFabrics[i]['fabric']
            # retrieve class name / item name
            foundObjectClass_ID = foundObjects[i]['class_id']
            foundObjectClassName = boundingBoxClassMap[str(foundObjectClass_ID)]
            # set bounding box dimensions
            foundObjectLeft = foundObjects[i]['left']
            foundObjectTop = foundObjects[i]['top']
            foundObjectWidth = foundObjects[i]['width']
            foundObjectHeight = foundObjects[i]['height']
            # print out what we found
            print("Found Object:", foundObject, "Class Name:", foundObjectClassName, "Confidence: ", boundingBoxConf, "Color: ", boundingBoxColor, "Fabric: ", boundingBoxFabric, "\n")
            # append to polly sentence
            polly_sentence += ", a " + str(boundingBoxColor) + " " + str(boundingBoxFabric) + " " + foundObjectClassName
            # append to objects JSON string
            objects_detected += '{"class": "' + str(foundObjectClassName) + '", "confidence": ' + str(boundingBoxConf) + ', "color": "' + str(boundingBoxColor) + '", "fabric": "' + str(boundingBoxFabric) +  \
            '", "left": ' + str(foundObjectLeft) + ', "top": ' +  str(foundObjectTop) + ', "width": ' + str(foundObjectWidth) + ', "height": ' + str(foundObjectHeight) + '}'
            if (i < numObjects-1):
                objects_detected += ', ' 

        objects_detected += ']'
        polly_sentence += "."
        # reformat objects_detected as JSON
        objects_detected = json.loads(objects_detected)


        # Print complete sentence for Amazon Polly
        # replace any underscores with spaces
        polly_sentence = polly_sentence.replace("_", " ")
        print(polly_sentence)
        
        # construct JSON for data payload for polly
        data_polly = {
            'fragment-number': int(fragmentNumber),
            'sentence' : polly_sentence
        }
        
        # Now send all data to the kinesis data stream (to be picked up for polly narration)
        put_to_stream(stream_out_polly_name, data_polly)
        
        # construct JSON for data payload for bounding box drawing on image
        data_imagebb = {
            'source-ref': sourceRef,
            'source-ref': originalImage,
            'fragment-number': int(fragmentNumber),
            'num-detected-objects': numObjects,
            'image-size': imageSize,
            'objects': objects_detected
        }
        
        # Now send all data to the kinesis data stream (to be picked up for bounding box drawing)
        put_to_stream(stream_out_imagebb_name, data_imagebb)
        
        print(str(numObjects) + " OBJECTS DETECTED")
        print("END OF EXECUTION")
        
    return 'Successfully processed {} records.'.format(len(event['Records']))
