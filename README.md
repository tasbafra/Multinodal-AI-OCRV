# Multinodal-AI-OCRV
Preventing Accidents through Advanced Communication Analysis
# ffmpeg
install ffmpeg from this link -> https://github.com/FFmpeg/FFmpeg
of zip archieve from the folder ffmpeg
# ffmpeg example
import subprocess

input_video = 'input_video.mp4'  #- use your path to the video file
output_audio = 'output_audio.mp3' #-  use your path to the audio file

#Команда ffmpeg для конвертации видео в аудио

command = ['ffmpeg', '-i', input_video, output_audio]

#Выполнение команды с помощью subprocess

subprocess.run(command, capture_output=True)

#Проверка вывода и ошибок

if subprocess.CompletedProcess.returncode == 0:
    print('Convertation succeeded')
else:
    print('Convertation failed')
# whisper
use this link to install whisper -> https://github.com/openai/whisper
or install archieve from the folder whisper
# whisper example
import whisper

def transcribe_audio_to_text(audio_path):
    #Загружаем модель, 'small' можно заменить на tiny, medium, base, large в зависимости от требуемого качества и скорости
    model = whisper.load_model("small")

    #Выполняем транскрибирование аудиофайла
    result = model.transcribe(audio_path)

    #Возвращаем транскрибированный текст
    return result['text']

#Пример использования функции
audio_file_path = "path/to/your/audio/file.mp3" #- путь к аудиофайлу
text = transcribe_audio_to_text(audio_file_path)
print(text)
#similarity





