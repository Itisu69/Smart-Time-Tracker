import cv2 as cv
import time

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"

camera = cv.VideoCapture(0)

camera.set(cv.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    print("Could'nt access webcam.")
    exit()

# YuNet Detector
detector = cv.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    (320,320),
    0.9,
    0.3,
    5000
)

last_detection_time = 0
detection_interval = 0.25  # 250 milliseconds

faces = None
status = "WE ARE CHECKING....."

# 30-second absence countdown
ABSENSE_THRESHOLD = 30

absent_since = None
display_status = "WE ARE CHECKING...."

# displays cam view in a separate window
while True:
    success, frame = camera.read()

    if not success:
        break

    height, width = frame.shape[:2]

    # Detecting faces
    current_time = time.monotonic()

    if current_time - last_detection_time >= detection_interval:

        # updating detector input dimension
        detector.setInputSize((width, height))

        _, faces = detector.detect(frame)

        face_present = faces is not None and len(faces) > 0
        
        last_detection_time = current_time

    if face_present is True:
        status = "PRESENT"
        absense_started_at = None
        
    elif face_present is False:
        
        if absense_started_at is None:
            absense_started_at = current_time
            
        absense_duration = current_time - absense_started_at
        
        if absense_duration >= ABSENSE_THRESHOLD:
            status = "AWAY - 30s"
        else:
            status = f"ABSENT - {int(absense_duration)}s"

    else:
        status = "WE ARE CHECKING...."
        
    # draws face rectangle
    if face_present and faces is not None:
        
        for face in faces:
            x, y, w, h = face[:4].astype(int)

        cv.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )
        
    # display status
    cv.putText(
        frame,
        status,
        (20, 40),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0) if status == "PRESENT" else (0, 0, 255),
        2
    )

    cv.imshow("Smart Time Tracker", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv.destroyAllWindows()