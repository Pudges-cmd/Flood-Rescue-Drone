import cv2
from ultralytics import YOLO

def main():
    # Load YOLO model (make sure 'yolo11n.pt' is in the same directory or provide the correct path)
    try:
        model = YOLO("yolo11n.pt")
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"Error initializing YOLO model: {e}")
        return

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Run YOLO detection
        results = model(frame)
        # Draw boxes for detected humans (class 0 is 'person' in COCO)
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Human Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()