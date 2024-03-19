import subprocess

input_video = 'input_video.mp4'
output_audio = 'output_audio.mp3'

# Команда ffmpeg для конвертации видео в аудио
command = ['ffmpeg', '-i', input_video, output_audio]

# Выполнение команды с помощью subprocess
subprocess.run(command, capture_output=True)

# Проверка вывода и ошибок
if subprocess.CompletedProcess.returncode == 0:
    print('Конвертация завершена успешно!')
else:
    print('Произошла ошибка при конвертации.')