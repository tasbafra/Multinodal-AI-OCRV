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
