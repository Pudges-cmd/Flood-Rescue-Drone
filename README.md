# Flood Rescue Drone Detection System I definately did not ask chatgpt to make this for me mhm yep

A comprehensive human and animal detection system designed for flood rescue operations using a Raspberry Pi 4B with SIM7600G-H 4G module and camera.

## Features

- **Multi-Object Detection**: Detects humans, dogs, and cats using YOLO11
- **GPS Integration**: Uses SIM7600G-H module for location tracking
- **SMS Alerts**: Sends real-time alerts with GPS coordinates
- **Object Tracking**: Prevents duplicate SMS alerts for the same person/animal
- **Photo Capture**: Automatically saves photos when persons are detected
- **Optimized Performance**: Designed for smooth real-time detection on Raspberry Pi

## Hardware Requirements

- Raspberry Pi 4B
- SIM7600G-H 4G module
- Raspberry Pi Camera v2 (or external webcam for testing)
- MicroSD card (32GB+ recommended)

## Software Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLO
- PySerial

## Installation

1. **Install system dependencies:**
   ```bash
   sudo apt-get update
   sudo apt-get upgrade
   sudo apt-get install python3-pip python3-opencv libatlas-base-dev
   ```

2. **Install Python packages:**
   ```bash
   pip3 install -r requirements.txt
   ```

3. **Connect SIM7600G-H module** to the Raspberry Pi via USB

4. **Connect camera** (Raspberry Pi Camera v2 or USB webcam)

## Usage

### Main Detection System
Run the complete flood rescue detection system:
```bash
python3 Main.py
```

### Human Detection Test
Test human detection functionality only:
```bash
python3 Yolo.py
```

## Configuration

### SMS Settings
Edit `Main.py` to configure:
- Phone number for alerts (line ~220)
- SMS cooldown period (line ~160)

### Detection Parameters
Adjust in `Main.py`:
- Confidence threshold (line ~175)
- IoU threshold (line ~175)
- Tracking parameters (line ~165)

## File Structure

```
├── Main.py              # Main detection system
├── Yolo.py              # Human detection test script
├── SMS.py               # SMS and GPS handling
├── requirements.txt     # Python dependencies
├── yolo11n.pt          # YOLO model (small)
├── yolo11x.pt          # YOLO model (large)
└── detection_photos/    # Saved detection photos
```

## SMS Alert Format

```
🚨 FLOOD RESCUE DRONE ALERT 🚨
Time: 2024-02-14 12:34:56
Location: 40.7128° N, 74.0060° W
Humans Detected: 2
Dogs Detected: 1
Cats Detected: 0
Status: Active Monitoring
```

## Performance Optimization

- Uses YOLO11x for better accuracy (falls back to YOLO11n if needed)
- Optimized camera settings for 30 FPS
- Efficient object tracking to prevent duplicate alerts
- 60-second SMS cooldown to prevent spam

## Troubleshooting

### Camera Issues
- Check camera connection
- Verify camera permissions
- Try different camera index (0, 1, 2)

### SMS Module Issues
- Verify SIM7600G-H connection
- Check SIM card status
- Ensure proper USB port assignment

### Performance Issues
- Reduce camera resolution
- Use YOLO11n instead of YOLO11x
- Close unnecessary applications

## Safety Notes

- This system is designed for emergency response
- Always verify detections before taking action
- GPS accuracy may vary based on conditions
- SMS delivery depends on cellular network coverage

## License

This project is designed for emergency response and rescue operations. 
