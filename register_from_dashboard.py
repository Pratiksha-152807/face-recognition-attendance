
import csv
import shutil
import sys
import time
from pathlib import Path

import cv2

# -------------------- SETTINGS --------------------

BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset"
STUDENTS_FILE = BASE / "students.csv"
CASCADE_FILE = BASE / "haarcascade_frontalface_default.xml"
MODEL_FILE = BASE / "models" / "face_model.yml"

IMAGES_REQUIRED = 50
DUPLICATE_THRESHOLD = 70

DATASET.mkdir(exist_ok=True)

# -------------------- STUDENT DETAILS --------------------

if len(sys.argv) < 3:
    print("ERROR: Student ID and name are required.")
    sys.exit(1)

student_id = sys.argv[1].strip()
student_name = " ".join(sys.argv[2:]).strip()

if not student_id or not student_name:
    print("ERROR: Enter a valid student ID and name.")
    sys.exit(1)

# Prevent unsafe folder names.
if (
    Path(student_id).name != student_id
    or student_id in {".", ".."}
    or "/" in student_id
    or "\\" in student_id
):
    print("ERROR: Invalid student ID.")
    sys.exit(1)

# -------------------- READ EXISTING STUDENTS --------------------

students = []
fieldnames = ["student_id", "name"]
id_column = "student_id"
name_column = "name"

if STUDENTS_FILE.exists():
    with open(STUDENTS_FILE, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames:
            fieldnames = reader.fieldnames

            def find_column(names, alternatives):
                for name in names:
                    if name and name.strip().lower() in alternatives:
                        return name
                return None

            id_column = find_column(
                fieldnames,
                {"student_id", "id", "roll_no", "roll number", "roll_number"}
            )
            name_column = find_column(
                fieldnames,
                {"name", "student_name", "student name", "full_name"}
            )

            if not id_column or not name_column:
                print(
                    "ERROR: Could not identify ID and name columns "
                    "in students.csv."
                )
                sys.exit(1)

            students = list(reader)

for student in students:
    if str(student.get(id_column, "")).strip() == student_id:
        print(f"ERROR: Student ID {student_id} is already registered.")
        sys.exit(1)

final_folder = DATASET / student_id
pending_folder = DATASET / f"pending_{student_id}"

if final_folder.exists():
    print(f"ERROR: Dataset folder for student {student_id} already exists.")
    sys.exit(1)

if pending_folder.exists():
    shutil.rmtree(pending_folder, ignore_errors=True)

# -------------------- LOAD FACE DETECTOR --------------------

if not CASCADE_FILE.exists():
    print(f"ERROR: Haar cascade file not found: {CASCADE_FILE}")
    sys.exit(1)

face_detector = cv2.CascadeClassifier(str(CASCADE_FILE))

if face_detector.empty():
    print("ERROR: Could not load the Haar cascade.")
    sys.exit(1)

# -------------------- LOAD DUPLICATE DETECTOR --------------------

recognizer = None
known_ids = {
    str(student.get(id_column, "")).strip()
    for student in students
}

if known_ids:
    if not MODEL_FILE.exists():
        print("ERROR: Trained face model is missing.")
        print("Train the existing students before registering another student.")
        sys.exit(1)

    if not hasattr(cv2, "face"):
        print("ERROR: OpenCV face module is unavailable.")
        print("Install opencv-contrib-python in your virtual environment.")
        sys.exit(1)

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(str(MODEL_FILE))
    except Exception as error:
        print(f"ERROR: Could not load the trained model: {error}")
        sys.exit(1)

# -------------------- OPEN WEBCAM --------------------

camera = None

for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
    try:
        camera = cv2.VideoCapture(0, backend)

        if camera.isOpened():
            ok, test_frame = camera.read()
            if ok and test_frame is not None:
                break

        camera.release()
        camera = None

    except Exception:
        if camera is not None:
            camera.release()
        camera = None

if camera is None or not camera.isOpened():
    print("ERROR: Could not open the webcam.")
    print("Close other camera apps and check Windows camera permissions.")
    sys.exit(1)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# -------------------- CAPTURE FACE IMAGES --------------------

pending_folder.mkdir(parents=True, exist_ok=True)

count = 0
duplicate = None
cancelled = False
camera_error = False
last_capture = 0.0

print("Webcam opened.")
print(f"Registering: {student_name} (ID: {student_id})")
print(f"Capture {IMAGES_REQUIRED} face images.")
print("Look at the camera and move your head slightly.")
print("Press Q in the camera window to cancel.")

try:
    while count < IMAGES_REQUIRED:
        ok, frame = camera.read()

        if not ok or frame is None:
            print("ERROR: Could not read a frame from the webcam.")
            camera_error = True
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (200, 200))

            # Check the face against existing registered students.
            if recognizer is not None:
                label, distance = recognizer.predict(face)
                matched_id = str(int(label))

                if matched_id in known_ids and distance < DUPLICATE_THRESHOLD:
                    duplicate = (matched_id, distance)
                    break

            now = time.time()

            # Save an image approximately every 0.15 seconds.
            if now - last_capture >= 0.15:
                image_path = pending_folder / f"{count + 1}.jpg"

                if cv2.imwrite(str(image_path), face):
                    count += 1
                    last_capture = now

            cv2.rectangle(
                frame, (x, y), (x + w, y + h), (0, 255, 0), 2
            )

            cv2.putText(
                frame,
                f"Images: {count}/{IMAGES_REQUIRED}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        if duplicate is not None:
            break

        cv2.imshow("Student Face Registration", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cancelled = True
            break

finally:
    camera.release()
    cv2.destroyAllWindows()

# -------------------- HANDLE REGISTRATION RESULT --------------------

if duplicate is not None:
    matched_id, distance = duplicate

    matched_name = matched_id
    for student in students:
        if str(student.get(id_column, "")).strip() == matched_id:
            matched_name = student.get(name_column, matched_id)
            break

    print("\nDUPLICATE FACE DETECTED!")
    print(f"Name: {matched_name}")
    print(f"Student ID: {matched_id}")
    print(f"Match distance: {distance:.2f}")
    print("Registration cancelled. No student record was saved.")

    shutil.rmtree(pending_folder, ignore_errors=True)
    sys.exit(1)

if cancelled:
    print("Registration cancelled by user.")
    shutil.rmtree(pending_folder, ignore_errors=True)
    sys.exit(1)

if camera_error or count < IMAGES_REQUIRED:
    print(f"ERROR: Only captured {count}/{IMAGES_REQUIRED} images.")
    print("Registration was not saved.")
    shutil.rmtree(pending_folder, ignore_errors=True)
    sys.exit(1)

# -------------------- SAVE STUDENT --------------------

try:
    # Do not overwrite an existing student folder.
    if final_folder.exists():
        print("ERROR: Student dataset folder already exists.")
        shutil.rmtree(pending_folder, ignore_errors=True)
        sys.exit(1)

    pending_folder.rename(final_folder)

    new_student = {column: "" for column in fieldnames}
    new_student[id_column] = student_id
    new_student[name_column] = student_name
    students.append(new_student)

    with open(STUDENTS_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(students)

    print("\nREGISTRATION SUCCESSFUL!")
    print(f"Name: {student_name}")
    print(f"Student ID: {student_id}")
    print(f"Images captured: {count}")
    print(f"Images saved in: {final_folder}")

except Exception as error:
    print(f"ERROR: Could not save registration: {error}")
    print("Check students.csv and the dataset folder before retrying.")
    sys.exit(1)
