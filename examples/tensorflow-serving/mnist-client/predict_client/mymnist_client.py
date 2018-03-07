#!/usr/bin/python
# This is directly adapted from:
# https://github.com/tobegit3hub/tensorflow_template_application/tree/master/python_predict_client
import os
import sys
import cv2
from PIL import Image
import operator
import numpy
#from keras.datasets.mnist import input_data  
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from predict_client import predict_pb2
from predict_client import prediction_service_pb2
from grpc.beta import implementations

tf.app.flags.DEFINE_string("host", "192.168.18.137", "gRPC server host")
tf.app.flags.DEFINE_integer("port", 9000, "gRPC server port")
tf.app.flags.DEFINE_string("model_name", "mnist", "TensorFlow model name")
tf.app.flags.DEFINE_integer("model_version", -1, "TensorFlow model version")
tf.app.flags.DEFINE_float("request_timeout", 10.0, "Timeout of gRPC request")
FLAGS = tf.app.flags.FLAGS



def main():
  host = FLAGS.host
  port = FLAGS.port
  model_name = FLAGS.model_name
  model_version = FLAGS.model_version
  request_timeout = FLAGS.request_timeout

  filename="example3.jpg"
  fullpath = os.path.join("/mydata/", filename)
  src_img = Image.open(fullpath, 'r')
  print('TF Processing source/reference image  %dx%d - %s.' % (src_img.size[0], src_img.size[1], src_img.format))
  src_img.show()
  
  # Create gRPC client and request
  channel = implementations.insecure_channel(host, port)
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = model_name
  if model_version > 0:
    request.model_spec.version.value = model_version
  data=cv2.imread(fullpath)
  print((data))
  flatdata=data.flatten()
  print((flatdata))
  inputimg=tf.contrib.util.make_tensor_proto(flatdata, shape=[1,28,28,1])
  print((inputimg))
  request.inputs['image'].CopyFrom(inputimg)  
  #request.inputs['image'].CopyFrom(features_tensor_proto)
  request.model_spec.signature_name = 'classify'
  

  # Send request
  result = stub.Predict(request, request_timeout)
  response = numpy.array(result.outputs['outputs'].float_val)
  prediction = numpy.argmax(response)

  print('prediction is %s ' % (prediction))


if __name__ == '__main__':
  main()