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
from netvlad_tf.srv import *
# import netvlad_tf.net_from_mat as nfm
# import scipy.io as scio
# import unittest
class NetVLADNode(object):
# class NetVLADNode(unittest.TestCase):
    def __init__(self):
        self._cv_bridge = CvBridge()
        
        # for inferencing from a different process/thread
        # self._graph = tf.get_default_graph()
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._image_batch = tf.placeholder(
                dtype=tf.float32, shape=[None, None, None, 3])
            self._net_out = nets.vgg16NetvladPca(self._image_batch)    
            self._saver = tf.train.Saver()

        
        config = tf.ConfigProto(device_count = {'GPU': 0})
        self._session = tf.Session(config = config, graph=self._graph)
        # with self._session as sess:
        #     self._saver.restore(sess, nets.defaultCheckpoint())
        self._saver.restore(self._session, nets.defaultCheckpoint())
        

    def run(self):
        # clears the current graph stack
        # tf.reset_default_graph()
        # self._test_net()
        # inference service handler
        s = rospy.Service('generate_descriptor',
                netvlad_result, self._handle_generate_descriptor)

        while not rospy.is_shutdown():
            rospy.spin()
    # def _test_net(self):
    #     inim = cv2.imread(nfm.exampleImgPath())
    #     inim = cv2.cvtColor(inim, cv2.COLOR_BGR2RGB)
    #     print("inim: ", inim.shape)
    #     batch = np.expand_dims(inim, axis=0)
    #     print("batch dim: ", batch.shape)

    #     #%% Generate TF results
    #     result = self._session.run(
    #                 self._net_out,
    #                 feed_dict={self._image_batch: batch})
    #     print(result.size)

    #     #%% Load Matlab results
    #     mat = scio.loadmat(nfm.exampleStatPath(),
    #                        struct_as_record=False, squeeze_me=True)
    #     mat_outs = mat['outs']

    #     #%% Compare final output
    #     out_diff = np.abs(mat_outs[-1] - result)
    #     self.assertLess(np.linalg.norm(out_diff), 0.0053)
    #     print('Error of final vector is %f' % np.linalg.norm(out_diff))

    def _handle_generate_descriptor(self, req):
        print("in handle generate descriptor")
        req_img = req.image
        np_image = self._cv_bridge.imgmsg_to_cv2(req_img, 'bgr8')
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        req_bb = req.boxes
        # crop the bounding boxes from the image and generate descriptors
        for bb in req_bb:
            x = bb.x_offset
            y = bb.y_offset
            w = bb.width
            h = bb.height
            bb_image = req_img[y:y+h, x:x+w]
            bb_image = np.expand_dims(bb_image, axis=0)
            result = self._session.run(
                    self._net_out,
                    feed_dict={self._image_batch: bb_image})
            print(result.size)
            resp.descriptors.append(result)            

        return resp

def main():
    rospy.init_node('netvlad_tf')

    node = NetVLADNode()
    node.run()


if __name__ == '__main__':
    main()