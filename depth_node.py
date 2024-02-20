from __future__ import absolute_import, division, print_function

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge
import cv2

from sensor_msgs.msg import Image
from interfaces_pkg.msg import Point2D
from interfaces_pkg.msg import BoundingBox2D
from interfaces_pkg.msg import Mask
from interfaces_pkg.msg import Detection
from interfaces_pkg.msg import DetectionArray

import argparse
import os
import sys

import glob
import numpy as np
import torch
import time

import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
print("fuck")

from torchvision import transforms, datasets
import networks
import layers as layers
import utils as utils 



SUB_TOPIC_NAME = "topic_raw_img"
#SUB_TOPIC_NAME = "topic_masking_img"
PUB_TOPIC_NAME = "topic_depth_img"
SHOW_IMAGE = True


TIMER = 0.1
QUE = 10

def parse_args():
    parser = argparse.ArgumentParser(
        description='for Monodepthv2 model.')

    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use', default="mono_640x192")
                        # choices=[
                        #     "mono_640x192",
                        #     "stereo_640x192",
                        #     "mono+stereo_640x192",
                        #     "mono_no_pt_640x192",
                        #     "stereo_no_pt_640x192",
                        #     "mono+stereo_no_pt_640x192",
                        #     "mono_1024x320",
                        #     "stereo_1024x320",
                        #     "mono+stereo_1024x320"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument('--width', type=int,
                        help='desired width', default=1080)
    parser.add_argument('--height', type=int,
                        help='image extension to search for in folder', default=720)
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()


class DepthNode(Node):
    def __init__(self, sub_topic=SUB_TOPIC_NAME, pub_topic=PUB_TOPIC_NAME, timer=TIMER, que=QUE):
        super().__init__('Depth')
        self.declare_parameter('sub_topic', sub_topic)
        self.declare_parameter('pub_topic', pub_topic)
        self.declare_parameter('timer', timer)
        self.declare_parameter('que', que)

        self.sub_topic = self.get_parameter('sub_topic').get_parameter_value().string_value
        self.pub_topic = self.get_parameter('pub_topic').get_parameter_value().string_value
        self.timer_period = self.get_parameter('timer').get_parameter_value().double_value
        self.que = self.get_parameter('que').get_parameter_value().integer_value

        image_qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, durability=QoSDurabilityPolicy.VOLATILE, depth=self.que)

        self.br = CvBridge()

        self.subscription = self.create_subscription(Image, self.sub_topic, self.callback, image_qos_profile)
        self.publisher_ = self.create_publisher(Image, self.pub_topic , self.que)
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # self.timer = self.create_timer(self.timer_period, self.timer_callback)


    def callback(self, msg):
        global args
        with torch.no_grad():
            print ("callback called")
            current_frame = self.br.imgmsg_to_cv2(msg)
            img = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
            original_img = img
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im_pil = pil.fromarray(img)
            # Load image and preprocess
            input_image = im_pil
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)
            print(disp_resized.type())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)
            open_cv_image = np.array(im)
            # Convert RGB to BGR
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            image_msg = self.br.cv2_to_imgmsg(open_cv_image, "rgb8")
            self.publisher_.publish(image_msg)

    def timer_callback(self):
        if not self.is_running:
            self.get_logger().info('Not published yet: "%s"' % self.sub_topic)
      

def main(args=None):
    rclpy.init(args=args)
    node = DepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
        pass
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    global args
    args = parse_args()
    width = args.width
    height = args.height

    assert args.model_name is not None, \
        "--model_name parameter"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    utils.download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")
    
    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()

    main()
