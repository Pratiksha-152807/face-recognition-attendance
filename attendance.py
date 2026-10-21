
import cv2
import csv
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent
MODEL = BASE / "models" / "face_model.yml"
STUDENTS = BASE / "students.csv"
ATTENDANCE = BASE / "attendance.csv"

if not MODEL.exists() or not STUDENTS.exists():
    print("Register students and train the model first.")
    raise SystemExit

with open(STUDENTS, newline="", encoding="utf-8") as f:
    students = {
        int(row["student_id"]): row["name"]
        for row in csv.DictReader(f)
    }

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(str(MODEL))


from pathlib import Path

cascade_path = (
    Path(__file__).parent
    / "haarcascade_frontalface_default.xml"
)

detector = cv2.CascadeClassifier(str(cascade_path))

if detector.empty():
    raise RuntimeError("Face detector XML could not be loaded.")


# Load attendance already recorded today
today = datetime.now().strftime("%Y-%m-%d")
marked = set()

if ATTENDANCE.exists():
    with open(ATTENDANCE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["date"] == today:
                marked.add(int(row["student_id"]))


def mark_attendance(student_id, name):
    # Prevent marking the same student twice today
    if student_id in marked:
        print(f"Already marked present today: {name}")
        return

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    file_exists = ATTENDANCE.exists()

    with open(ATTENDANCE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists or ATTENDANCE.stat().st_size == 0:
            writer.writerow([
                "student_id", "name", "date", "time"
            ])

        writer.writerow([student_id, name, date, time])

    marked.add(student_id)
    print(f"Attendance marked: {name} at {time}")


camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Webcam not found.")
    raise SystemExit

print("Press Q to exit.")

try:
    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray, 1.2, 5, minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            face = cv2.resize(gray[y:y+h, x:x+w],
                              (200, 200))

            label, confidence = recognizer.predict(face)

            # LBPH distance is lower for a closer match.
            # Tune this threshold using your own test images.
            if label in students and confidence < 70:
                name = students[label]
                color = (0, 255, 0)

                if label not in marked:
                    mark_attendance(label, name)

                status = "Present"
            else:
                name = "Unknown"
                color = (0, 0, 255)
                status = "Not recognized"

            cv2.rectangle(frame, (x, y), (x+w, y+h),
                          color, 2)
            cv2.putText(
                frame, f"{name} - {status}",
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, color, 2
            )

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
