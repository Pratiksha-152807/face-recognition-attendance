
import cv2
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent
DATASET = BASE / "dataset"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

faces = []
labels = []

for folder in DATASET.iterdir():
    if not folder.is_dir() or not folder.name.isdigit():
        continue

    student_id = int(folder.name)

    for image_path in folder.glob("*.jpg"):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is not None:
            image = cv2.resize(image, (200, 200))
            faces.append(image)
            labels.append(student_id)

if len(faces) < 2 or len(set(labels)) < 2:
    print("Register at least two students before training.")
    raise SystemExit

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels, dtype=np.int32))

recognizer.save(str(MODELS / "face_model.yml"))

print("Training completed successfully!")
print("Total face images:", len(faces))
print("Model saved in models/face_model.yml")
