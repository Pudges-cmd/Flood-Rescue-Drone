from ultralytics import YOLO
import cv2
from SMS import SMSHandler
import time
import numpy as np

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

class HumanTracker:
    def __init__(self, iou_threshold=0.5, max_frames=30):
        self.tracked_humans = []  # List of [box, frames_since_seen]
        self.iou_threshold = iou_threshold
        self.max_frames = max_frames
        
    def update(self, current_boxes):
        # Update frames since seen for existing tracks
        for track in self.tracked_humans:
            track[1] += 1
            
        # Remove old tracks
        self.tracked_humans = [track for track in self.tracked_humans 
                             if track[1] < self.max_frames]
        
        # Match current detections with existing tracks
        new_humans = []
        for box in current_boxes:
            is_new = True
            for track in self.tracked_humans:
                if calculate_iou(box, track[0]) > self.iou_threshold:
                    track[1] = 0  # Reset frames since seen
                    is_new = False
                    break
            if is_new:
                new_humans.append(box)
                self.tracked_humans.append([box, 0])
        
        return new_humans

def main():
    # Initialize YOLO model
    try:
        model = YOLO("yolo11n.pt")
    except Exception as e:
        print(f"Error initializing YOLO model: {e}")
        return
    
    # Initialize SMS handler
    sms = SMSHandler()
    if not sms.connect():
        print("Failed to connect to SMS module")
        return
        
    # Initialize camera
    camera = cv2.VideoCapture(0)  # Use 0 for default camera, adjust if needed
    if not camera.isOpened():
        print("Failed to open camera")
        sms.disconnect()
        return
    
    # Set camera properties
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize tracker
    tracker = HumanTracker(iou_threshold=0.5, max_frames=30)
    
    last_alert_time = 0
    alert_interval = 30  # Minimum seconds between alerts
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Run YOLO detection
            results = model(frame)
            
            # Process results
            for result in results:
                boxes = result.boxes
                classes = result.names
                
                # Get human boxes
                human_boxes = []
                for box in boxes:
                    if classes[int(box.cls)] == 'person':
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        human_boxes.append([x1, y1, x2, y2])
                
                # Update tracker and get new humans
                new_humans = tracker.update(human_boxes)
                
                # Draw boxes and labels for all humans
                for box in human_boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Human", 
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 255, 0), 2)
                
                # Print and send alert only for new humans
                if new_humans:
                    print(f"New human(s) detected: {len(new_humans)}")
                    
                    # Send alert if enough time has passed
                    current_time = time.time()
                    if (current_time - last_alert_time) >= alert_interval:
                        phone_number = "+639514343942"  # Philippines number
                        if sms.send_detection_alert(phone_number, len(new_humans)):
                            print(f"Alert sent: {len(new_humans)} new humans detected")
                            last_alert_time = current_time
                        else:
                            print("Failed to send alert")
            
            # Display the frame
            # cv2.imshow('Human Detection', frame) # Commented out for headless operation
            
            # Break loop on 'q' press
            # if cv2.waitKey(1) & 0xFF == ord('q'): # Commented out for headless operation
            #     break
                
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        # Cleanup
        camera.release()
        # cv2.destroyAllWindows() # Commented out for headless operation
        sms.disconnect()

if __name__ == "__main__":
    main()
