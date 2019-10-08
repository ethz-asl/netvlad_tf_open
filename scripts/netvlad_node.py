#!/usr/bin/env python
import os
import numpy as np
import sys
# np.set_printoptions(threshold=sys.maxsize)

import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import RegionOfInterest

import tensorflow as tf
import netvlad_tf.nets as nets
from netvlad_tf.msg import Descriptor
from netvlad_tf.srv import *

import colorsys
import random
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon

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
        req_zip = zip(req.boxes, req.class_numbers, req.instance_numbers)
        # Create Response
        resp = netvlad_resultResponse()
        bb_masks = []
        # crop the bounding boxes from the image and generate descriptors
        for bb, class_num, inst_num in req_zip:
            x = bb.x_offset
            y = bb.y_offset
            w = bb.width
            h = bb.height
            bb_image = np_image[y:y+h, x:x+w, :]
            # mask region
            if req.use_mask:
                np_mask = self._cv_bridge.imgmsg_to_cv2(req.mask, 'bgr8')
                mask_class, mask_instance, mask_confidence = cv2.split(np_mask)
                bb_class = mask_class[y:y+h, x:x+w]
                bb_instance = mask_instance[y:y+h, x:x+w]
                bb_class_mask = np.where(bb_class == class_num, 1, 0).astype(np.uint8)
                bb_instance_mask = np.where(bb_instance == inst_num, 1, 0).astype(np.uint8)
                bb_mask = (np.multiply(bb_class_mask, bb_instance_mask)).astype(np.uint8)
                bb_masks.append(bb_mask)
                bb_image_masked = np.ones(bb_image.shape, dtype = np.uint8)
                for c in range(3): 
                    bb_image_masked[:,:,c] = np.where(bb_mask == 1, bb_image[:, :, c], 0)
                ## for debugging
                # bb_image_masked = cv2.cvtColor(bb_image_masked, cv2.COLOR_RGB2BGR)
                # cv2.imshow('bb_image_masked', bb_image_masked)
                # cv2.waitKey(0)
                bb_image = bb_image_masked
            bb_image = np.expand_dims(bb_image, axis=0)    
            result = self._session.run(
                    self._net_out,
                    feed_dict={self._image_batch: bb_image})
            descriptor = Descriptor()
            descriptor.descriptor = list(result.flatten())
            resp.descriptors.append(descriptor)

        ## for debugging
        # visualization
        # _, ax = plt.subplots(1, figsize=(16,16))
        # height, width = np_image.shape[:2]
        # ax.set_ylim(height + 10, -10)
        # ax.set_xlim(-10, width + 10)
        # ax.axis('off')
        # N = len(req.boxes)
        # colors = random_colors(N)
        # masked_image = np_image.astype(np.uint8).copy()
        
        # for i in range(N):
        #     color = colors[i]
        #     x1 = req.boxes[i].x_offset
        #     y1 = req.boxes[i].y_offset
        #     x2 = req.boxes[i].x_offset + req.boxes[i].width
        #     y2 = req.boxes[i].y_offset + req.boxes[i].height

        #     # add bounding boxes
        #     p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
        #                         alpha=0.7, linestyle="dashed",
        #                         edgecolor=color, facecolor='none')
        #     ax.add_patch(p)

        #     # add masks
        #     mask = np.zeros((masked_image.shape[0], masked_image.shape[1]), dtype=np.uint8)
        #     mask[y1:y2, x1:x2] = bb_masks[i]
        #     masked_image = apply_mask(masked_image, mask, color)
        # ax.imshow(masked_image.astype(np.uint8))
        # plt.show()

        return resp

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    # hsv = [(i / N, 1, brightness) for i in range(N)]
    hsv = [(i / float(N), 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def main():
    rospy.init_node('netvlad_tf')

    node = NetVLADNode()
    node.run()


if __name__ == '__main__':
    main()