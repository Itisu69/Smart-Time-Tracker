import cv2 as cv

camera = cv.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not access webcam.")
    exit()

while True:
    success, frame = camera.read()

    if not success:
        print("Error: Could not read camera frame.")
        break

    cv.imshow("Webcam Test", frame)

    # Press Q to quit
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv.destroyAllWindows()