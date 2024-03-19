# Multinodal-AI-OCRV
Preventing Accidents through Advanced Communication Analysis
# ffmpeg
install ffmpeg from this link -> https://github.com/FFmpeg/FFmpeg
of zip archieve from the folder ffmpeg
# ffmpeg example
```
 import subproces

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
```
# whisper
use this link to install whisper -> https://github.com/openai/whisper
or install archieve from the folder whisper
# whisper example
```
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



