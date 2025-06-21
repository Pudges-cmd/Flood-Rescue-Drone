from ultralytics import YOLO
import cv2
from SMS import SMSHandler
import time
import numpy as np
import os
from datetime import datetime

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

class DetectionTracker:
    def __init__(self, iou_threshold=0.6, max_frames=120, max_distance=150):
        self.tracked_objects = []  # List of [box, frames_since_seen, object_id, last_seen_time, object_type]
        self.iou_threshold = iou_threshold
        self.max_frames = max_frames  # Increased to 4 seconds at 30fps
        self.max_distance = max_distance  # Maximum pixel distance for matching
        self.next_object_id = 1
        self.sms_sent_ids = set()  # Track which object IDs have received SMS
        self.object_first_seen = {}  # Track when each object was first detected
        self.photos_saved = set()  # Track which person IDs have had photos saved
        
    def update(self, current_detections):
        """
        current_detections: list of [box, object_type, confidence]
        """
        current_time = time.time()
        
        # Update frames since seen for existing tracks
        for track in self.tracked_objects:
            track[1] += 1
            
        # Remove old tracks that haven't been seen for too long
        self.tracked_objects = [track for track in self.tracked_objects 
                             if track[1] < self.max_frames]
        
        # Match current detections with existing tracks
        matched_tracks = set()
        new_objects = []
        new_object_ids = []
        
        for box, obj_type, confidence in current_detections:
            best_match = None
            best_score = 0
            best_track_idx = -1
            
            # Find best matching existing track
            for i, track in enumerate(self.tracked_objects):
                if i in matched_tracks:
                    continue
                    
                # Only match same object types
                if track[4] != obj_type:
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
                # This is a new object
                object_id = self.next_object_id
                self.next_object_id += 1
                
                # Add new track
                new_track = [box, 0, object_id, current_time, obj_type]
                self.tracked_objects.append(new_track)
                new_objects.append((box, obj_type, confidence))
                new_object_ids.append(object_id)
                
                # Record first detection time
                self.object_first_seen[object_id] = current_time
                
                print(f"New {obj_type} detected with ID: {object_id}")
        
        return new_objects, new_object_ids
    
    def get_unsent_object_ids(self, object_ids):
        """Return object IDs that haven't received SMS yet"""
        return [oid for oid in object_ids if oid not in self.sms_sent_ids]
    
    def mark_sms_sent(self, object_ids):
        """Mark object IDs as having received SMS"""
        self.sms_sent_ids.update(object_ids)
        for oid in object_ids:
            print(f"SMS sent for object ID: {oid}")
    
    def get_current_counts(self):
        """Get count of currently tracked objects by type"""
        counts = {'person': 0, 'dog': 0, 'cat': 0}
        for track in self.tracked_objects:
            obj_type = track[4]
            if obj_type in counts:
                counts[obj_type] += 1
        return counts
    
    def get_tracking_info(self):
        """Get detailed tracking information for debugging"""
        counts = self.get_current_counts()
        info = {
            'total_tracked': len(self.tracked_objects),
            'sms_sent_count': len(self.sms_sent_ids),
            'person_count': counts['person'],
            'dog_count': counts['dog'],
            'cat_count': counts['cat'],
            'object_ids': [track[2] for track in self.tracked_objects],
            'sms_sent_ids': list(self.sms_sent_ids)
        }
        return info
    
    def should_save_photo(self, object_id):
        """Check if we should save a photo for this person (only once per person)"""
        return object_id not in self.photos_saved
    
    def mark_photo_saved(self, object_id):
        """Mark that a photo has been saved for this person"""
        self.photos_saved.add(object_id)

def save_detection_photo(frame, object_id, object_type):
    """Save a photo when a person is detected"""
    try:
        # Create photos directory if it doesn't exist
        photos_dir = "detection_photos"
        if not os.path.exists(photos_dir):
            os.makedirs(photos_dir)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{photos_dir}/{object_type}_ID{object_id}_{timestamp}.jpg"
        
        # Save the frame
        cv2.imwrite(filename, frame)
        print(f"Photo saved: {filename}")
        return filename
    except Exception as e:
        print(f"Error saving photo: {e}")
        return None

def main():
    # Initialize YOLO model (using larger model for better accuracy)
    try:
        model = YOLO("yolo11x.pt")  # Using larger model for better detection
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"Error initializing YOLO model: {e}")
        print("Falling back to yolo11n.pt")
        try:
            model = YOLO("yolo11n.pt")
            print("YOLO11n model loaded successfully")
        except Exception as e2:
            print(f"Error loading YOLO11n model: {e2}")
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
    
    # Set camera properties for optimal performance
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)  # Set to 30 FPS
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for real-time
    print("Camera initialized successfully")
    
    # Initialize tracker with optimized parameters
    tracker = DetectionTracker(iou_threshold=0.6, max_frames=120, max_distance=150)
    
    frame_count = 0
    start_time = time.time()
    last_sms_time = 0
    sms_cooldown = 60  # Minimum 60 seconds between SMS alerts
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            
            # Run YOLO detection with optimized parameters
            results = model(frame, conf=0.5, iou=0.45, verbose=False)  # Lower confidence for better detection
            
            # Process results
            current_detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                    
                classes = result.names
                
                # Get human and animal boxes
                for box in boxes:
                    class_name = classes[int(box.cls)]
                    confidence = float(box.conf[0])
                    
                    # Only track humans, dogs, and cats
                    if class_name in ['person', 'dog', 'cat']:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        current_detections.append(([x1, y1, x2, y2], class_name, confidence))
            
            # Update tracker and get new objects
            new_objects, new_object_ids = tracker.update(current_detections)
            
            # Draw boxes and labels for all detected objects
            for box, obj_type, confidence in current_detections:
                x1, y1, x2, y2 = box
                
                # Find the object ID for this box
                object_id = None
                for track in tracker.tracked_objects:
                    if calculate_iou(box, track[0]) > 0.4:  # Relaxed for display
                        object_id = track[2]
                        break
                
                # Choose color based on object type
                if obj_type == 'person':
                    color = (0, 255, 0)  # Green for humans
                elif obj_type == 'dog':
                    color = (255, 0, 0)  # Blue for dogs
                else:  # cat
                    color = (0, 0, 255)  # Red for cats
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with object type and ID
                label = f"{obj_type.capitalize()}"
                if object_id:
                    label += f" ID:{object_id}"
                    # Mark if SMS was sent
                    if object_id in tracker.sms_sent_ids:
                        label += " (SMS Sent)"
                
                cv2.putText(frame, label, 
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, color, 2)
            
            # Handle SMS alerts and photo saving for new objects
            if new_objects:
                unsent_object_ids = tracker.get_unsent_object_ids(new_object_ids)
                
                if unsent_object_ids:
                    current_time = time.time()
                    
                    # Check SMS cooldown
                    if current_time - last_sms_time >= sms_cooldown:
                        print(f"New objects detected: {len(unsent_object_ids)} (IDs: {unsent_object_ids})")
                        
                        # Get current counts for SMS
                        counts = tracker.get_current_counts()
                        
                        # Send SMS for new objects
                        phone_number = "+639514343942"  # Philippines number
                        
                        if sms_connected:
                            success = sms.send_detection_alert(phone_number, counts['person'], counts['dog'], counts['cat'])
                            if success:
                                print(f"SMS Alert sent for {counts['person']} persons, {counts['dog']} dogs, {counts['cat']} cats")
                                tracker.mark_sms_sent(unsent_object_ids)
                                last_sms_time = current_time
                            else:
                                print("Failed to send SMS alert")
                        else:
                            # Simulate SMS for testing
                            print(f"[SIMULATED SMS] Alert: {counts['person']} persons, {counts['dog']} dogs, {counts['cat']} cats detected")
                            print(f"[SIMULATED SMS] To: {phone_number}")
                            print(f"[SIMULATED SMS] Object IDs: {unsent_object_ids}")
                            tracker.mark_sms_sent(unsent_object_ids)
                            last_sms_time = current_time
                    else:
                        print(f"SMS cooldown active, skipping alert for {len(unsent_object_ids)} new objects")
                
                # Save photos for new persons
                for i, (box, obj_type, confidence) in enumerate(new_objects):
                    if obj_type == 'person':
                        object_id = new_object_ids[i]
                        if tracker.should_save_photo(object_id):
                            save_detection_photo(frame, object_id, obj_type)
                            tracker.mark_photo_saved(object_id)
            
            # Print tracking statistics every 60 frames (2 seconds at 30fps)
            if frame_count % 60 == 0:
                tracking_info = tracker.get_tracking_info()
                print(f"Tracking Stats - Persons: {tracking_info['person_count']}, "
                      f"Dogs: {tracking_info['dog_count']}, Cats: {tracking_info['cat_count']}, "
                      f"Total SMS sent: {tracking_info['sms_sent_count']}")
            
            # Display the frame
            cv2.imshow('Flood Rescue Drone Detection', frame)
            
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
        print(f"Total unique objects detected: {tracker.next_object_id - 1}")
        print(f"Persons: {tracking_info['person_count']}")
        print(f"Dogs: {tracking_info['dog_count']}")
        print(f"Cats: {tracking_info['cat_count']}")
        print(f"SMS alerts sent: {len(tracker.sms_sent_ids)}")
        print(f"Photos saved: {len(tracker.photos_saved)}")
        
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
