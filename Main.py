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

def calculate_center_distance(box1, box2):
    """Calculate distance between box centers"""
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

class HumanTracker:
    def __init__(self, iou_threshold=0.7, max_frames=90, max_distance=100):
        self.tracked_humans = []  # List of [box, frames_since_seen, person_id, last_seen_time]
        self.iou_threshold = iou_threshold
        self.max_frames = max_frames  # Increased to 3 seconds at 30fps
        self.max_distance = max_distance  # Maximum pixel distance for matching
        self.next_person_id = 1
        self.sms_sent_ids = set()  # Track which person IDs have received SMS
        self.person_first_seen = {}  # Track when each person was first detected
        
    def update(self, current_boxes):
        current_time = time.time()
        
        # Update frames since seen for existing tracks
        for track in self.tracked_humans:
            track[1] += 1
            
        # Remove old tracks that haven't been seen for too long
        removed_tracks = []
        self.tracked_humans = [track for track in self.tracked_humans 
                             if track[1] < self.max_frames]
        
        # Match current detections with existing tracks using improved matching
        matched_tracks = set()
        new_humans = []
        new_person_ids = []
        
        for box in current_boxes:
            best_match = None
            best_score = 0
            best_track_idx = -1
            
            # Find best matching existing track
            for i, track in enumerate(self.tracked_humans):
                if i in matched_tracks:
                    continue
                    
                # Calculate IoU and distance scores
                iou_score = calculate_iou(box, track[0])
                distance = calculate_center_distance(box, track[0])
                
                # Combined scoring: IoU must be above threshold and distance reasonable
                if iou_score > self.iou_threshold and distance < self.max_distance:
                    # Weighted score favoring IoU
                    combined_score = iou_score * 0.7 + (1 - distance / self.max_distance) * 0.3
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_match = track
                        best_track_idx = i
            
            if best_match is not None:
                # Update existing track
                best_match[0] = box  # Update box position
                best_match[1] = 0    # Reset frames since seen
                best_match[3] = current_time  # Update last seen time
                matched_tracks.add(best_track_idx)
            else:
                # This is a new person
                person_id = self.next_person_id
                self.next_person_id += 1
                
                # Add new track
                new_track = [box, 0, person_id, current_time]
                self.tracked_humans.append(new_track)
                new_humans.append(box)
                new_person_ids.append(person_id)
                
                # Record first detection time
                self.person_first_seen[person_id] = current_time
                
                print(f"New person detected with ID: {person_id}")
        
        return new_humans, new_person_ids
    
    def get_unsent_person_ids(self, person_ids):
        """Return person IDs that haven't received SMS yet"""
        return [pid for pid in person_ids if pid not in self.sms_sent_ids]
    
    def mark_sms_sent(self, person_ids):
        """Mark person IDs as having received SMS"""
        self.sms_sent_ids.update(person_ids)
        for pid in person_ids:
            print(f"SMS sent for person ID: {pid}")
    
    def get_current_person_count(self):
        """Get count of currently tracked persons"""
        return len(self.tracked_humans)
    
    def get_tracking_info(self):
        """Get detailed tracking information for debugging"""
        info = {
            'total_tracked': len(self.tracked_humans),
            'sms_sent_count': len(self.sms_sent_ids),
            'person_ids': [track[2] for track in self.tracked_humans],
            'sms_sent_ids': list(self.sms_sent_ids)
        }
        return info

def main():
    # Initialize YOLO model
    try:
        model = YOLO("yolo11n.pt")
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"Error initializing YOLO model: {e}")
        print("Make sure yolo11n.pt is in the same directory")
        return
    
    # Initialize SMS handler
    sms = SMSHandler()
    sms_connected = sms.connect()
    if not sms_connected:
        print("Failed to connect to SMS module - continuing without SMS")
        print("SMS alerts will be simulated in console")
        
    # Initialize camera
    camera = cv2.VideoCapture(0)  # Use 0 for default camera, adjust if needed
    if not camera.isOpened():
        print("Failed to open camera")
        if sms_connected:
            sms.disconnect()
        return
    
    # Set camera properties
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Camera initialized successfully")
    
    # Initialize tracker with improved parameters
    tracker = HumanTracker(iou_threshold=0.7, max_frames=90, max_distance=100)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            
            # Run YOLO detection
            results = model(frame)
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                    
                classes = result.names
                
                # Get human boxes
                human_boxes = []
                for box in boxes:
                    if classes[int(box.cls)] == 'person':
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        human_boxes.append([x1, y1, x2, y2])
                
                # Update tracker and get new humans
                new_humans, new_person_ids = tracker.update(human_boxes)
                
                # Draw boxes and labels for all humans
                for i, box in enumerate(human_boxes):
                    x1, y1, x2, y2 = box
                    
                    # Find the person ID for this box
                    person_id = None
                    for track in tracker.tracked_humans:
                        if calculate_iou(box, track[0]) > 0.5:  # Relaxed for display
                            person_id = track[2]
                            break
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label with person ID
                    label = f"Person"
                    if person_id:
                        label += f" ID:{person_id}"
                        # Mark if SMS was sent
                        if person_id in tracker.sms_sent_ids:
                            label += " (SMS Sent)"
                    
                    cv2.putText(frame, label, 
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 255, 0), 2)
                
                # Handle SMS alerts for new persons only
                if new_person_ids:
                    unsent_person_ids = tracker.get_unsent_person_ids(new_person_ids)
                    
                    if unsent_person_ids:
                        print(f"New persons detected: {len(unsent_person_ids)} (IDs: {unsent_person_ids})")
                        
                        # Send SMS for new persons
                        phone_number = "+639514343942"  # Philippines number
                        
                        if sms_connected:
                            success = sms.send_detection_alert(phone_number, len(unsent_person_ids))
                            if success:
                                print(f"SMS Alert sent for {len(unsent_person_ids)} new persons")
                                tracker.mark_sms_sent(unsent_person_ids)
                            else:
                                print("Failed to send SMS alert")
                        else:
                            # Simulate SMS for testing
                            print(f"[SIMULATED SMS] Alert: {len(unsent_person_ids)} new persons detected")
                            print(f"[SIMULATED SMS] To: {phone_number}")
                            print(f"[SIMULATED SMS] Person IDs: {unsent_person_ids}")
                            tracker.mark_sms_sent(unsent_person_ids)
                
                # Print tracking statistics every 30 frames
                if frame_count % 30 == 0:
                    tracking_info = tracker.get_tracking_info()
                    print(f"Tracking Stats - Currently tracked: {tracking_info['total_tracked']}, "
                          f"Total SMS sent: {tracking_info['sms_sent_count']}")
            
            # Display the frame (uncomment for visual debugging)
            cv2.imshow('Human Detection', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping detection...")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        # Print final statistics
        tracking_info = tracker.get_tracking_info()
        print(f"\nFinal Statistics:")
        print(f"Total unique persons detected: {tracker.next_person_id - 1}")
        print(f"SMS alerts sent: {len(tracker.sms_sent_ids)}")
        print(f"Person IDs that received SMS: {list(tracker.sms_sent_ids)}")
        
        # Cleanup
        camera.release()
        cv2.destroyAllWindows()
        if sms_connected:
            sms.disconnect()
        
        # Calculate runtime
        runtime = time.time() - start_time
        print(f"Total runtime: {runtime:.1f} seconds")
        print(f"Average FPS: {frame_count / runtime:.1f}")

if __name__ == "__main__":
    main()
