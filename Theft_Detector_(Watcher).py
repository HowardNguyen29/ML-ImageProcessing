import cv2
import time

cap = cv2.VideoCapture(1)

ret, first_frame = cap.read()
gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output_TrungHau_CartoonifyFaces.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))


start_detection_time = 0
delay_duration = 5 


reset_baseline_time = 0
reset_baseline_interval = 10

while True:
    
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if time.time() - reset_baseline_time > reset_baseline_interval:
        # Reset the first frame
        ret, first_frame = cap.read()
        gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        if time.time() - reset_baseline_time < 5:
            a = True

        reset_baseline_time = time.time()

    abs_diff = cv2.absdiff(gray_frame, gray_first_frame)
    _, thresholded = cv2.threshold(abs_diff, 130, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "UNSAFE", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    start_detection_time = time.time()
    
    cv2.imshow('Original Frame', frame)
    out.write(frame)
    cv2.imshow('Frame with Rectangles', thresholded)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
