
import cv2
import csv
from pathlib import Path

BASE = Path(__file__).parent
DATASET = BASE / "dataset"
STUDENTS = BASE / "students.csv"

DATASET.mkdir(exist_ok=True)

student_id = input("Enter student ID: ").strip()
name = input("Enter student name: ").strip()

if not student_id.isdigit() or not name:
    print("Enter a numeric ID and a valid name.")
    raise SystemExit

student_folder = DATASET / student_id
student_folder.mkdir(exist_ok=True)

# Store student details
students = {}

if STUDENTS.exists():
    with open(STUDENTS, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            students[row["student_id"]] = row["name"]

students[student_id] = name

with open(STUDENTS, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["student_id", "name"])
    for sid, student_name in students.items():
        writer.writerow([sid, student_name])

camera = cv2.VideoCapture(0)

from pathlib import Path

cascade_path = (
    Path(__file__).parent
    / "haarcascade_frontalface_default.xml"
)

detector = cv2.CascadeClassifier(str(cascade_path))

if detector.empty():
    raise RuntimeError("Face detector XML could not be loaded.")



count = 0
print("Look at the camera. Press Q to stop.")

while count < 50:
    success, frame = camera.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        count += 1
        image_path = student_folder / f"face_{count}.jpg"
        cv2.imwrite(str(image_path), face)

        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (0, 255, 0), 2)
        cv2.putText(frame, f"Images: {count}/50",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        break

    cv2.imshow("Register Student", frame)
    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print(f"Captured {count} face images for {name}.")
