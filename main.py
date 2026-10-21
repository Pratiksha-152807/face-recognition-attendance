
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Webcam not found!")
    exit()

while True:
    success, frame = camera.read()

    if not success:
        print("Cannot read webcam")
        break

    cv2.imshow("Face Recognition Attendance", frame)

    # Press Q to close
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
