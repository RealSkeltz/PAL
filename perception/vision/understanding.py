import ollama
import cv2
import base64

MODEL = "qwen3.5:2b"

def interpret_image(image):
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "Analyse the image in terms of threat level. Scan for weapons, hostility or other signs of danger.",
                "images": [numpy_to_base64(image)]  # ✅ attach image here
            }
        ]
    )

    return response.message.content

def numpy_to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")