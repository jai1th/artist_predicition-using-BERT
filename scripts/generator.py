import subprocess
import predict_artist
import time

def generate_lyrics(prompt):
    result = subprocess.run(
        ["ollama", "run", "gemma3:4b"],
        input=prompt.encode(),
        capture_output=True
    )
    return result.stdout.decode()

def main(prompt):
    start = time.time()
    gemma_lyrics = generate_lyrics(prompt)
    print("Time taken to generate lyrics: ", time.time() - start)
    print("\nGemma generated lyrics: ", gemma_lyrics)
    start = time.time()
    prediction = predict_artist.main(gemma_lyrics)
    print("Time taken to predict artist: ", time.time() - start)
    return prediction