from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf
import io
# import google.generativeai as genai
# import os

# genai.configure(api_key=os.environ['AIzaSyDb9IyUyl-xEF7m4YB2vOELyd5HNPmnz-Y'])

# model = genai.GenerativeModel(name='gemini-1.5-flash')

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5173/detect",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../models/Version_2")

CLASS_NAMES = ["Black Sigatoka", "Bract Mosaic Virus", "Healthy", "Insect Pest", "Moko", "Panama", "Yellow Sigatoka"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

@app.post("/predict") 
async def predict(image: bytes = File(...)):
    image_obtained = Image.open(io.BytesIO(image))
    image_resized = image_obtained.resize((512, 512))
    image_bytes = np.array(image_resized)
    img_batch = np.expand_dims(image_bytes, 0)
    
    predictions = MODEL.predict(img_batch)
    val = []
    confi = []
    for i in predictions:
        val.append(np.argmax(i[0]))
        confi.append(np.max(i[0]))
    max_confidence = max(confi)
    index = confi.index(max_confidence)
    confidence = round(100 * max_confidence, 2)
    predicted_class = CLASS_NAMES[val[index]]
    # response = model.generate_content('Give cure for {predicted_class}')
    # print(response.text)
    return {
        'class': predicted_class,
        # 'confidence': response
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
