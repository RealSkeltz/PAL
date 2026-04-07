import ollama
import cv2
import base64

### QWEN APPROACH

MODEL = "qwen3.5:0.8b"

def interpret_image(image, context):
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": f"Describe the image in one concise sentence. There is a {context} in the image.",
                "images": [numpy_to_base64(image)]  # ✅ attach image here
            }
        ]
    )

    return response.message.content

def numpy_to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")