"""
 Lambda function for looking up most recent identification and responding back to Lex for voice response.
 Author: Matt White <matt.white@berkeley.edu>
 Date: 12/13/2021
"""

import json
import os
import logging
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
import time
import calendar
from boto3 import Session
from boto3 import resource
import codecs

region_name = "us-west-2"
bucket_name = "descriptiveworld-demo-audio"

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


""" --- Helpers to build responses which match the structure of the necessary dialog actions --- """

def close(session_attributes, fulfillment_state, message):
    response = {
        'sessionAttributes': session_attributes,
        'dialogAction': {
            'type': 'Close',
            'fulfillmentState': fulfillment_state,
            'message': message
        }
    }

    return response


""" --- Functions that control the bot's behavior --- """

# main function for identifying item
def identify_item(intent_request):
    """
    lookup the current.txt file in s3 and return its contents
    """
    
    source = intent_request['invocationSource']
    output_session_attributes = intent_request['sessionAttributes'] if intent_request['sessionAttributes'] is not None else {}

    # setup session
    session = Session(region_name=region_name)
    
    # s3 bucket assignments
    s3 = boto3.client('s3')
        
    # text filename settings
    current_text_filename = "current.txt"
    
    data = s3.get_object(Bucket=bucket_name, Key=current_text_filename)
    contents = data['Body'].read()

    # Read the current.txt audio
    response_text = contents.decode("utf-8")
    logger.debug('loaded current.txt contents from S3')

    return close(
        output_session_attributes,
        'Fulfilled',
        {
            'contentType': 'PlainText',
            'content': response_text
        }
    )


""" --- Intents --- """


def dispatch(intent_request):
    """
    Called when the user specifies an intent for this bot.
    """

    logger.debug('dispatch userId={}, intentName={}'.format(intent_request['userId'], intent_request['currentIntent']['name']))

    intent_name = intent_request['currentIntent']['name']

    # Dispatch to your bot's intent handlers
    if intent_name == 'IdentifyItem':
        return identify_item(intent_request)
    raise Exception('Intent with name ' + intent_name + ' not supported')


""" --- Main handler --- """


def lambda_handler(event, context):
    """
    Route the incoming request based on intent.
    The JSON body of the request is provided in the event slot.
    """
    # By default, treat the user request as coming from the America/New_York time zone.
    os.environ['TZ'] = 'America/New_York'
    time.tzset()
    logger.debug('event.bot.name={}'.format(event['bot']['name']))

    return dispatch(event)
