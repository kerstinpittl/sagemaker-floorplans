import base64
import json
from io import BytesIO
import requests
import numpy as np


def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API

    Args:
        data (obj): the request data stream
        context (Context): an object containing request and configuration details

    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    if context.request_content_type == 'application/x-npy':  # for numpy array tensor
        # invoke endpoint with serialized numpy array as request body, the TensorFlow Serving SignatureDef should assume input dtype: [1, H, W, C=3]
        image_npy = np.load(BytesIO(data.read()))
        return json.dumps({"inputs": image_npy.tolist()})
    
    else:
        _return_error(417, 'Unsupported content type "{}"'.format(context.request_content_type or 'Unknown')) 

def _npy_dumps(data):
    """
    Serialized a numpy array into a stream of npy-formatted bytes.
    """
    buffer = BytesIO()
    np.save(buffer, data)
    return buffer.getvalue()

def output_handler(response, context):
    """Post-process TensorFlow Serving output before it is returned to the client.

    Args:
        response (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details

    Returns:
        (bytes, string): data to return to client, response content type
    """
    
    print("output status code: " )
    print(response.status_code)
    if response.status_code != 200:
        print(response.content)
        _return_error(response.status_code, response.content.decode('utf-8'))
        
    response_content_type = context.accept_header
    if response_content_type == 'application/x-npy':
        return _npy_dumps(response.content), 'application/x-npy'
    else:
        
        prediction = response.content
        return prediction, response_content_type


def _return_error(code, message):
    print(message)
    raise ValueError('Error: {}, {}'.format(str(code), message))