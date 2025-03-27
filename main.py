#!/usr/bin/python3

import numpy as np
import cv2
from cv_bridge import CvBridge
from matplotlib import pyplot as plt
import rosbag
import yaml
import time

def detect_ORB(img: cv2.Mat):
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(img, None)
    ef_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
    return kp, des, ef_img

def match_ORB(des1: np.ndarray, des2: np.ndarray, threshold: float):
    bf = cv2.BFMatcher(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    max_distance = matches[-1].distance
    threshold_distance = threshold * max_distance
    filtered_matches = [m for m in matches if m.distance < threshold_distance]
    return filtered_matches
    
def main():
    with open('config.yaml') as f:
        fs = yaml.safe_load(f)
    
    bridge = CvBridge()
    cv2.namedWindow('ORB Extracted Frames', cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow('ORB Extracted Frames', 710, 683)

    with rosbag.Bag(fs['bag_dir'], 'r') as bag:
        image_topic = fs['img_topic']
        messages = [msg for _, msg, _ in bag.read_messages(topics = [image_topic])]

        for i in range(len(messages) - 1):
            img1 = bridge.imgmsg_to_cv2(messages[i], desired_encoding='bgr8')
            ts1 = messages[i].header.stamp.to_sec()
            kp1, des1, ef_img1 = detect_ORB(img1)
            
            img2 = bridge.imgmsg_to_cv2(messages[i+1], desired_encoding='bgr8')
            ts2 = messages[i+1].header.stamp.to_sec()
            kp2, des2, ef_img2 = detect_ORB(img2)

            matches = match_ORB(des1, des2, fs['threshold'])

            matched_image = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            raw_image = np.hstack((img1, img2))
            processed_image = np.hstack((ef_img1, ef_img2))
            final_image = np.vstack((np.vstack((raw_image, processed_image)), matched_image))

            try:
                cv2.imshow('ORB Extracted Frames', final_image)
                cv2.waitKey(1)
                delay = ts2 - ts1
                time.sleep(max(0, delay))
            except Exception as e:
                print(f"Error converting image: {e}")

    return

if __name__ == '__main__':
    main()
