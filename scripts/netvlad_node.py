#!/usr/bin/env python
import os
import numpy as np

import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import RegionOfInterest

import tensorflow as tf
import netvlad_tf.nets as nets
from netvlad_tf.msg import Descriptor
from netvlad_tf.srv import *

class NetVLADNode(object):
    def __init__(self):
        self._cv_bridge = CvBridge()
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._image_batch = tf.placeholder(
                dtype=tf.float32, shape=[None, None, None, 3])
            self._net_out = nets.vgg16NetvladPca(self._image_batch)    
            self._saver = tf.train.Saver()        
        config = tf.ConfigProto(device_count = {'GPU': 0})
        self._session = tf.Session(config = config, graph=self._graph)
        self._saver.restore(self._session, nets.defaultCheckpoint())

    def run(self):
        # inference service handler
        s = rospy.Service('generate_descriptor',
                netvlad_result, self._handle_generate_descriptor)

        while not rospy.is_shutdown():
            rospy.spin()

    def _handle_generate_descriptor(self, req):
        print("in handle generate descriptor")
        req_img = req.image
        np_image = self._cv_bridge.imgmsg_to_cv2(req_img, 'bgr8')
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        req_bb = req.boxes
        # Create Response
        resp = netvlad_resultResponse()
        # crop the bounding boxes from the image and generate descriptors
        for bb in req_bb:
            x = bb.x_offset
            y = bb.y_offset
            w = bb.width
            h = bb.height
            bb_image = np_image[y:y+h, x:x+w]
            bb_image = np.expand_dims(bb_image, axis=0)
            result = self._session.run(
                    self._net_out,
                    feed_dict={self._image_batch: bb_image})
            descriptor = Descriptor()
            descriptor.descriptor = list(result.flatten())
            resp.descriptors.append(descriptor)
        return resp

def main():
    rospy.init_node('netvlad_tf')

    node = NetVLADNode()
    node.run()


if __name__ == '__main__':
    main()