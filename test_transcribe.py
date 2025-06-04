import requests

audio_path = "Hey Nutan Dekha Dik Aar Bar With Lyrics  Swagatalakshmi Dasgupta.mp3"
url = "http://127.0.0.1:8000/transcribe"

with open(audio_path, "rb") as f:
    files = {"file": (audio_path, f, "audio/wav")}
    resp = requests.post(url, files=files)
    print(resp.json())
