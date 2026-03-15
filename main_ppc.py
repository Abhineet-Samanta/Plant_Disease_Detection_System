import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# -------------------------
# LOAD MODELS
# -------------------------

POTATO_MODEL = tf.keras.models.load_model(
    r"C:\Users\Admin\PycharmProjects\TensorFlow\Model\model_v1.keras"
)

PEPPER_MODEL = tf.keras.models.load_model(
    r"C:\Users\Admin\PycharmProjects\TensorFlow\Bell_pepper_Disease_Classification\model_v1.keras"
)

CORN_MODEL = tf.keras.models.load_model(
    r"C:\Users\Admin\PycharmProjects\TensorFlow\Corn_Disease_Classification\model_v1.keras"
)

# -------------------------
# CLASS LABELS
# -------------------------

POTATO_CLASSES = ["EARLY BLIGHT", "LATE BLIGHT", "HEALTHY"]

PEPPER_CLASSES = ["BACTERIAL SPOT", "HEALTHY"]

CORN_CLASSES = [
    "COMMON RUST",
    "GRAY LEAF SPOT",
    "NORTHERN LEAF BLIGHT",
    "HEALTHY"
]


# -------------------------
# IMAGE PREPROCESS
# -------------------------

def read_file_as_image(data):
    image = np.array(Image.open(BytesIO(data)))
    return image


# -------------------------
# SERVER CHECK
# -------------------------

@app.get("/ping")
async def ping():
    return "SERVER RUNNING"


# -------------------------
# POTATO PREDICTION
# -------------------------

@app.post("/predict/potato")
async def predict_potato(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)

    prediction = POTATO_MODEL.predict(img_batch)

    predicted_class = POTATO_CLASSES[np.argmax(prediction[0])]
    confidence = float(np.max(prediction[0]))

    return {
        "plant": "potato",
        "class": predicted_class,
        "confidence": confidence
    }


# -------------------------
# BELL PEPPER PREDICTION
# -------------------------

@app.post("/predict/pepper")
async def predict_pepper(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)

    prediction = PEPPER_MODEL.predict(img_batch)

    predicted_class = PEPPER_CLASSES[np.argmax(prediction[0])]
    confidence = float(np.max(prediction[0]))

    return {
        "plant": "bell_pepper",
        "class": predicted_class,
        "confidence": confidence
    }


# -------------------------
# CORN PREDICTION
# -------------------------

@app.post("/predict/corn")
async def predict_corn(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)

    prediction = CORN_MODEL.predict(img_batch)

    predicted_class = CORN_CLASSES[np.argmax(prediction[0])]
    confidence = float(np.max(prediction[0]))

    return {
        "plant": "corn",
        "class": predicted_class,
        "confidence": confidence
    }


# -------------------------
# RUN SERVER
# -------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8006)