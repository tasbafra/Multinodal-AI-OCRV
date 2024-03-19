import argparse
import sys
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

from joblib import dump, load

import torch
import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data impor  t Dataset, DataLoader

# class_names = ['stop', 'green light', 'up', 'down', 'conducting']
class_names = [' Initiate a full stop', 'Receive authority to depart', 'Increase throttle', ' Reduce throttle', 'Coordinate carriage approach']

#dtree = load('dtree.joblib')

class MLP_KP(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(MLP_KP, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.num_features = num_features

    def forward(self, x):
        # print(x.shape)
        x = x.view(-1, self.num_features)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

num_features = 99
num_classes = 5

model = MLP_KP(num_features=num_features, num_classes=num_classes)#.to(device)
model.load_state_dict(torch.load('mlp_keypoints_model.pth'))
model.eval()

#@markdown To better demonstrate the Pose Landmarker API, we have created a set of visualization tools that will be used in this colab. These will draw the landmarks on a detect person, as well as the expected connections between those markers.
def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def descriptor_from_detection(detection_result):
    landmarks_list = detection_result.pose_landmarks[0]
    cc = []
    for ll in landmarks_list:
        cc.append(ll.x)
        cc.append(ll.y)
        cc.append(ll.z)
    return np.array(cc)

def parse_args():	
    parser=argparse.ArgumentParser(description="video pipeline")
    parser.add_argument("video_file")
    args=parser.parse_args()
    return args


# Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options, output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

def get_phase_from_video(filename):
    print(">>> Processing:", filename)

    cap = cv2.VideoCapture(filename)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)//5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)//5)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    frame_count = 0
    frames_keypoints = []
    frames_actions = []

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count < total_count // 2:
            continue

        frame = cv2.resize(frame, (width, height))
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Detect pose landmarks from the input image.
        detection_result = detector.detect(mp_frame)
        if len(detection_result.pose_landmarks) == 0:
            continue

        coords = descriptor_from_detection(detection_result)
        frames_keypoints.append(coords)

        coords = descriptor_from_detection(detection_result)
        if len(coords) == 0:
           continue
        # xx = np.expand_dims(coords, axis=0)
        # yy = dtree.predict(xx)[0]
        #### MLP version
        data2_predict = torch.FloatTensor(coords)
        data2_predict.unsqueeze_(0)
        outputs = model(data2_predict)
        _, predicted = torch.max(outputs.data, 1)
        predicted_idx = predicted.squeeze_().cpu().numpy()        

        # action_name = class_names[yy]
        frames_actions.append(predicted_idx)

    cap.release()

    # print(frames_actions)
    class_labels = np.array(frames_actions)
    mean_label = np.mean(class_labels)
    print(mean_label)
    label_forall = int(mean_label+0.5)

    action_name = class_names[label_forall]
    # print(action_name)

    m = nn.Softmax(dim=1)
    proba = m(outputs)
    # print(proba.squeeze().detach().cpu().numpy())

    return action_name, proba.squeeze().detach().cpu().numpy()


if __name__ == '__main__':
    inputs=parse_args()
    # print(inputs.video_file)
    phrase, proba = get_phase_from_video(inputs.video_file)
    print(phrase)
    print(proba)
