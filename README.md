# Multinodal-AI-OCRV
Preventing Accidents through Advanced Communication Analysis
# ffmpeg
install ffmpeg from this link -> https://github.com/FFmpeg/FFmpeg
of zip archieve from the folder ffmpeg
# ffmpeg example
```
import subprocesses

 input_video = 'input_video.mp4' #- use your path to the video file
 output_audio = 'output_audio.mp3' #- use your path to the audio file

 #FFmpeg command to convert video to audio
 command = ['ffmpeg', '-i', input_video, output_audio]

 #Execute a command using subprocess
 subprocess.run(command, capture_output=True)

 #Check output and errors
 if subprocess.CompletedProcess.returncode == 0:
     print('Conversion succeeded')
 else:
     print('Conversion failed')
```
# whisper
use this link to install whisper -> https://github.com/openai/whisper
or install archieve from the folder whisper
# whisper example
```
 import whisper

 def transcribe_audio_to_text(audio_path):
     #Load the model, 'small' can be replaced with tiny, medium, base, large depending on the required quality and speed
    model = whisper.load_model("small")

  #Transcribe the audio file
    result = model.transcribe(audio_path)

    #Returning the transcribed text
    return result['text']

#Example of using the function
audio_file_path = "path/to/your/audio/file.mp3" #- path to the audio file
text = transcribe_audio_to_text(audio_file_path)
print(text)

```
# similarity
install sentence_transformers from this link -> https://github.com/UKPLab/sentence-transformers
or from the folder similarity
# similarity_example
```
from sentence_transformers import SentenceTransformer, util

def get_similarity(input_sentence, default_sentences):
    model = SentenceTransformer('egorishti/sentence-similarity-v1-based-2.0')

    # Encode the input and default sentences
    input_embedding = model.encode(input_sentence, convert_to_tensor=True)
    default_embeddings = model.encode(default_sentences, convert_to_tensor=True)

    # Calculate the cosine similarity between the input and default sentences
    similarities = util.pytorch_cos_sim(input_embedding, default_embeddings)

    return similarities

# Example usage
audio_path = 'gest1.wav'  # Change this to the path of your audio file
input_command = transcribe_audio_with_whisper(audio_path)
# input_command = "Initiate A full stop"
default_sentences = ["Begin a complete halt", "stop entirely", "Apply brakes", "Initiate a comprehensive cessation", "Start bringing it to a standstill", "Go faster", "Green light to move", "You can move", "Increase throttle", "You have the authority to depart"] # you can use your examples

input_command = transcribe_audio_with_whisper(audio_path)
similarities = get_similarity(input_command, default_sentences)
print(similarities)  # This will show a list of similarities with each default sentence
```
# main_code
before starting the code, download mip_keypoints_model and pose_landmarker from the data folder
to start the web app open the link in the folder data
```
import os

import time

import torch
import torchaudio
import numpy as np
import gradio as gr
import pandas as pd
import whisper
import subprocess
from sentence_transformers import SentenceTransformer, util

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

default_sentences_more = ["Begin a complete halt", "stop entirely", "Apply brakes", "Initiate a comprehensive cessation", "Start bringing it to a standstill", "Go faster", "Green light to move", "You can move", "Increase throttle", "You have the authority to depart"]
default_sentences = [' Initiate a full stop.', 'Receive authority to depart.', 'Increase throttle.', ' Reduce throttle.', 'Coordinate carriage approach.']

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

    action_name = default_sentences[label_forall]
    # print(action_name)

    m = nn.Softmax(dim=1)
    proba = m(outputs)
    # print(proba.squeeze().detach().cpu().numpy())

    return action_name, proba.squeeze().detach().cpu().numpy(), label_forall

def get_similarity(input_sentence, default_sentences):
    model = SentenceTransformer('egorishti/sentence-similarity-v1-based-2.0')

    # Encode the input and default sentences
    input_embedding = model.encode(input_sentence, convert_to_tensor=True)
    default_embeddings = model.encode(default_sentences, convert_to_tensor=True)

    # Calculate the cosine similarity between the input and default sentences
    similarities = util.pytorch_cos_sim(input_embedding, default_embeddings)

    return similarities


class MultimodalAI(gr.Blocks):
    def __init__(self, title, css=None, theme=None, sr=16000, compute_type="float16", lang="english", task="transcribe", whisper_name = "tiny"):
        super().__init__(
            title=title,
            css=css,
            theme=theme
        )
        self.sr = sr
        
        # whisper path
        self.model = whisper.load_model(whisper_name, download_root="./Models")
        self.whisper_options = {"language": lang, "task": task}
        
        # Gradio Inputs
        self.videofile = gr.Video(label='Load video', interactive=True, include_audio=True)
        self.input_components = [self.videofile]
        
        # Gradio Outputs
        self.asr_result = gr.Textbox(label="Transcribed text")
        self.text_similarity = gr.Textbox(label="Text Similarity")
        self.video_action = gr.Textbox(label="Video Action")
        self.video_similarity = gr.Textbox(label="Video Similarity")
        self.resolver = gr.Image(label="Result", show_download_button=False, show_label=False)
        self.output_components = [self.asr_result, self.text_similarity, self.video_action, self.video_similarity, self.resolver]
        
        with self:
            gr.Markdown('<p style="font-size: 2.5em; text-align: center; margin-bottom: 1rem"><span style="font-family:Source Sans Pro; color:black"> Multimodal AI </span></p>')
            
            with gr.Row():
                with gr.Column():
                    self.videofile.render()
                with gr.Column():
                    self.asr_result.render()
                    self.text_similarity.render()
                    self.video_action.render()
                    self.video_similarity.render()
                    self.resolver.render()
            with gr.Row():
                run_btn = gr.Button('Launch', variant='primary', elem_id='run_btn')
                clear_btn = gr.ClearButton(
                            self.input_components, value="Clear"
                        )
                
            run_btn.click(self.predict_result, self.input_components, self.output_components)      
            


    def _video2audio(self, videofile):
        filename, ext = os.path.splitext(videofile)
        subprocess.call(["ffmpeg", "-y", "-i", videofile, "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", f"{filename}.wav"], 
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT)
        return f"{filename}.wav"


    def _preprocess_record(self, waveform, sr):
        """Подготовка записи (разные микрофоны пишут в разном формате)"""
        if waveform.dtype == 'int16':
            waveform = torch.tensor(waveform.astype(np.float32, order='C') / 32768.0).T
        elif waveform.dtype == 'int32':
            waveform = torch.tensor(waveform.astype(np.float32, order='C') / 2147483648.0).T
        
        # добавление размерности    
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            gr.Warning('Number of channels > 1. Only first channel will be processed.')
            waveform = waveform[0].unsqueeze(0)
        # ресэмплинг
        if sr != self.sr:
            waveform = torchaudio.transforms.Resample(sr, self.sr)(waveform)
        return waveform
    
    
    def transcribe(self, waveform):
        transcript = self.model.transcribe(waveform, **self.whisper_options)["text"]
        return transcript
        
    def predict_result(self,
                videofile=None,
        ):
        if videofile is not None:
            audiofilename = self._video2audio(videofile)
            waveform, sr = torchaudio.load(audiofilename)

        start0 = time.time()
        waveform = self._preprocess_record(waveform, sr)
        waveform = waveform[0].numpy()
        print("Audio duration:", len(waveform) / self.sr)
        
        print("Preprocessing:", time.time() - start0)

        phrase_video, proba_video, video_idx = get_phase_from_video(videofile)
        print(">> Video:", phrase_video)
        print(">> Video proba:", proba_video)

        start = time.time()
        asr_result = self.transcribe(waveform)
        print("Transcription:", time.time() - start)
        
        similarities = get_similarity(asr_result, default_sentences)
        np_sim = similarities.squeeze().detach().cpu().numpy()
        
        max_simularity_idx = np.argmax(np_sim)
        print(np_sim)  # This will show a list of similarities with each default sentence
        
        text_df = np_sim

        proba_video *= 1000
        proba_video = proba_video.astype(np.int32)
        proba_video = proba_video.astype(np.float32)
        proba_video /= 1000

        video_df = proba_video

        if max_simularity_idx == video_idx: # and np_sim[max_simularity_idx] > 0.7 and video_df[video_idx] > 0.75:
            image = "good.png"
        else:
            image = "bad.png"

        return asr_result, text_df, phrase_video, video_df,  image

if __name__ == "__main__":
    theme = gr.themes.Base(
        primary_hue="red",
        secondary_hue="red",
    )
    
    demo = MultimodalAI(
        title="Multimodal AI",
        theme=theme,
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7000,
        share=True,
        show_error=True,
        # auth = ("superman", "superman123"),
        # auth_message = "Введите логин и пароль",
        # debug=True,
    )
```


