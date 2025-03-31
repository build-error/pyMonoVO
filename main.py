#!/usr/bin/python3

import numpy as np
import cv2
from cv_bridge import CvBridge
import rosbag
import yaml
import time
from scipy.spatial.transform import Rotation
import csv

def detect_ORB(img: cv2.Mat):
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(img, None)
    ef_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
    return kp, des, ef_img

def match_ORB(kp1: tuple, kp2: tuple, des1: np.ndarray, des2: np.ndarray, threshold: float):
    bf = cv2.BFMatcher(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    threshold_distance = threshold * matches[-1].distance
    good, pts1, pts2 = [], [], []
    for m in matches:
        if m.distance < threshold_distance:
            good.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    
    return good, pts1, pts2

def camera_pose_estimation(pts1: np.array, pts2: np.array, fs: dict):
    fx, fy, cx, cy = fs['intrinsics']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    # F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    # H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3)
    E, _ = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K)
    _, R, t, _ = cv2.recoverPose(E, points1=pts1, points2=pts2, cameraMatrix=K)
    return R, t


def main():
    with open('config.yaml') as f:
        fs = yaml.safe_load(f)
    
    bridge = CvBridge()
    cv2.namedWindow('ORB Extracted Frames', cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow('ORB Extracted Frames', 710, 683)

    f = open('cameraTrajectory.txt', 'w')

    with rosbag.Bag(fs['bag_dir'], 'r') as bag:
        image_topic = fs['img_topic']
        messages = [msg for _, msg, _ in bag.read_messages(topics = [image_topic])]

        for i in range(len(messages) - 1):
            img1 = bridge.imgmsg_to_cv2(messages[i], desired_encoding='bgr8')
            ts1 = messages[i].header.stamp.to_sec()
            
            img2 = bridge.imgmsg_to_cv2(messages[i+1], desired_encoding='bgr8')
            ts2 = messages[i+1].header.stamp.to_sec()
            
            kp1, des1, ef_img1 = detect_ORB(img1)
            kp2, des2, ef_img2 = detect_ORB(img2)
            matches, pts1 , pts2 = match_ORB(kp1, kp2, des1, des2, fs['threshold'])
            
            try:
                R, t = camera_pose_estimation(pts1, pts2, fs)
                q = Rotation.from_matrix(R).as_quat()
                f.write(f"{ts1} {t[0][0]} {t[1][0]} {t[2][0]} {float(q[0])}, {float(q[1])} {float(q[2])} {float(q[3])}\n")


            except Exception as e:
                print(f"Error from camera_pose_estimation: {e}")

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
