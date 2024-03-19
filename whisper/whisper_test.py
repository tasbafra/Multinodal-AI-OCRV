import whisper

def transcribe_audio_to_text(audio_path):
    # Загружаем модель, 'small' можно заменить на tiny, medium, base, large в зависимости от требуемого качества и скорости
    model = whisper.load_model("small")

    # Выполняем транскрибирование аудиофайла
    result = model.transcribe(audio_path)

    # Возвращаем транскрибированный текст
    return result['text']

# Пример использования функции
audio_file_path = "path/to/your/audio/file.mp3"
text = transcribe_audio_to_text(audio_file_path)
print(text)
