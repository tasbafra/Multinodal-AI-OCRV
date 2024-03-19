import os
import time
import torch
import torchaudio
import numpy as np
import gradio as gr
import pandas as pd
import whisper
import subprocess


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
        self.slider1 = gr.Slider(label="whisper confidence", minimum=0, maximum=1, step=0.1)
        self.slider2 = gr.Slider(label="simularity confidence", minimum=0, maximum=1, step=0.1)
        self.slider3 = gr.Slider(label="mediapipe pose confidence", minimum=0, maximum=1, step=0.1)
        self.text_commands = gr.Textbox(label="Commands", lines=5)
        self.input_components = [self.videofile, self.slider1, self.slider2, self.slider3, self.text_commands]
        
        # Gradio Outputs
        self.asr_result = gr.Textbox(label="Transcribed text")
        self.text_similarity = gr.Dataframe(label="1")
        self.video_similarity = gr.Dataframe(label="2")
        self.resolver = gr.Image(label="Result", show_download_button=False, show_label=False)
        self.output_components = [self.asr_result, self.text_similarity, self.video_similarity, self.resolver]
        
        
        
        with self:
            gr.Markdown('<p style="font-size: 2.5em; text-align: center; margin-bottom: 1rem"><span style="font-family:Source Sans Pro; color:black"> Multimodal AI </span></p>')
            
            with gr.Row():
                with gr.Column():
                    self.videofile.render()
                    self.slider1.render()
                    self.slider2.render()
                    self.slider3.render()
                    self.text_commands.render()
                with gr.Column():
                    self.asr_result.render()
                    self.text_similarity.render()
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
                slider1=None,
                slider2=None,
                slider3=None,
                text_commands=None
        ):
        if videofile is not None:
            audiofilename = self._video2audio(videofile)
            waveform, sr = torchaudio.load(audiofilename)

        start0 = time.time()
        waveform = self._preprocess_record(waveform, sr)
        waveform = waveform[0].numpy()
        print("Audio duration:", len(waveform) / self.sr)
        
        print("Preprocessing:", time.time() - start0)
        
        start = time.time()
        asr_result = self.transcribe(waveform)
        print("Transcription:", time.time() - start)
        
        text_df = pd.DataFrame()
        video_df = pd.DataFrame()
        image = "/home/experiments/Models/22.png"
        
        return asr_result, text_df, video_df, image

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