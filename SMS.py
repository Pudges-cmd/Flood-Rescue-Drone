import serial
import time
import json
from datetime import datetime
import serial.tools.list_ports
import platform
import os

class SMSHandler:
    def __init__(self, port=None, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.is_raspberry_pi = platform.system() == 'Linux' and os.path.exists('/proc/device-tree/model')
        self.last_sms_time = 0
        self.min_sms_interval = 30  # Minimum seconds between SMS messages
        self.gps_data = None
        
    def find_sim7600_port(self):
        """Find the port for SIM7600G-H module on Windows or Raspberry Pi"""
        if self.is_raspberry_pi:
            # On Raspberry Pi, typically /dev/ttyUSB0 or /dev/ttyACM0
            possible_ports = ['/dev/ttyUSB2', '/dev/ttyACM0']
            for port in possible_ports:
                if os.path.exists(port):
                    return port
            return None
        else:
            # Windows port detection
            ports = list(serial.tools.list_ports.comports())
            for port in ports:
                if "USB Serial Device" in port.description:
                    return port.device
            return None
        
    def connect(self):
        try:
            # If no port specified, try to find it automatically
            if not self.port:
                self.port = self.find_sim7600_port()
                if not self.port:
                    print("Could not automatically find SIM7600G-H module")
                    if not self.is_raspberry_pi:
                        print("Available ports:")
                        for port in serial.tools.list_ports.comports():
                            print(f"- {port.device}: {port.description}")
                    else:
                        print("Please check if the module is properly connected")
                    return False
                print(f"Found SIM7600G-H on port: {self.port}")
            
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for module to initialize
            
            # Test if the module is responding
            response = self.send_command('AT')
            if not response or 'OK' not in response:
                print("Module not responding to AT command")
                self.disconnect()
                return False
                
            # Initialize cellular network
            if not self.initialize_cellular():
                print("Failed to initialize cellular network")
                return False
                
            # Initialize GPS
            if not self.initialize_gps():
                print("Failed to initialize GPS")
                return False
                
            print(f"Connected to SIM7600G-H on {self.port}")
            return True
        except serial.SerialException as e:
            print(f"Serial port error: {e}")
            return False
        except Exception as e:
            print(f"Error connecting to SIM7600G-H: {repr(e)}")
            return False
            
    def initialize_cellular(self):
        """Initialize cellular network connection"""
        try:
            # Check SIM card status
            response = self.send_command('AT+CPIN?', wait_time=2)
            if not response or '+CPIN: READY' not in response:
                print("SIM card not ready")
                return False
                
            # Check network registration
            response = self.send_command('AT+CREG?', wait_time=2)
            if not response or '+CREG: 0,1' not in response:
                print("Not registered to network")
                return False
                
            # Check signal quality
            response = self.send_command('AT+CSQ', wait_time=2)
            if not response:
                print("Could not check signal quality")
                return False
                
            print("Cellular network initialized successfully")
            return True
        except serial.SerialException as e:
            print(f"Serial port error: {e}")
            return False
        except Exception as e:
            print(f"Error connecting to SIM7600G-H: {repr(e)}")
            return False
            
    def initialize_gps(self):
        """Initialize GPS functionality"""
        try:
            # Enable GPS
            if not self.send_command('AT+CGNSPWR=1', wait_time=2):
                return False
                
            # Set GPS NMEA output
            if not self.send_command('AT+CGNSSEQ="RMC"', wait_time=2):
                return False
                
            return True
        except Exception as e:
            print(f"Error initializing GPS: {e}")
            return False
            
    def get_gps_location(self):
        """Get current GPS location"""
        try:
            # Request GPS data
            response = self.send_command('AT+CGNSINF', wait_time=2)
            if not response:
                return None
                
            # Parse GPS data
            # Format: +CGNSINF: 1,1,20240214123456.000,40.7128,-74.0060,0.00,0.0,1,,1.1,1.5,0.9,,11,6,,,42,,
            parts = response.split(',')
            if len(parts) >= 5 and parts[0] == '+CGNSINF: 1':
                lat = float(parts[3])
                lon = float(parts[4])
                return f"{lat:.4f}° N, {lon:.4f}° W"
            return None
        except Exception as e:
            print(f"Error getting GPS location: {e}")
            return None
            
    def disconnect(self):
        if self.serial:
            try:
                # Turn off GPS before disconnecting
                self.send_command('AT+CGNSPWR=0', wait_time=2)
                self.serial.close()
                print("Disconnected from SIM7600G-H")
            except Exception as e:
                print(f"Error during disconnect: {e}")
            
    def send_command(self, command, wait_time=1):
        if not self.serial:
            return None
        try:
            self.serial.write((command + '\r\n').encode())
            time.sleep(wait_time)
            response = self.serial.read(self.serial.in_waiting).decode()
            return response
        except Exception as e:
            print(f"Error sending command: {e}")
            return None
        
    def send_sms(self, phone_number, message):
        if not self.serial:
            return False
            
        # Check if enough time has passed since last SMS
        current_time = time.time()
        if current_time - self.last_sms_time < self.min_sms_interval:
            wait_time = self.min_sms_interval - (current_time - self.last_sms_time)
            print(f"Waiting {wait_time:.1f} seconds before sending next SMS...")
            time.sleep(wait_time)
            
            
        try:
            # Set SMS text mode
            if not self.send_command('AT+CMGF=1', wait_time=2):
                return False
            
            # Set character set to GSM
            if not self.send_command('AT+CSCS="GSM"', wait_time=2):
                return False
            
            # Send SMS
            self.send_command(f'AT+CMGS="{phone_number}"', wait_time=2)
            self.serial.write(message.encode() + chr(26).encode())
            time.sleep(5)  # Increased wait time for SMS sending
            
            response = self.serial.read(self.serial.in_waiting).decode()
            if "OK" in response:
                self.last_sms_time = time.time()
                return True
            return False
        except Exception as e:
            print(f"Error sending SMS: {e}")
            return False
        
    def send_detection_alert(self, phone_number, human_count, dog_count=0, cat_count=0):
        # Get current GPS location
        gps_location = self.get_gps_location()
        if not gps_location:
            gps_location = "GPS data unavailable"
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"🚨 FLOOD RESCUE DRONE ALERT 🚨\n"
        message += f"Time: {timestamp}\n"
        message += f"Location: {gps_location}\n"
        message += f"Humans Detected: {human_count}\n"
        if dog_count > 0:
            message += f"Dogs Detected: {dog_count}\n"
        if cat_count > 0:
            message += f"Cats Detected: {cat_count}\n"
        message += f"Status: Active Monitoring"
        
        return self.send_sms(phone_number, message)

# Example usage
if __name__ == "__main__":
    # Initialize SMS handler
    sms = SMSHandler()
    
    # Connect to the module
    if sms.connect():
        # Example phone number and data
        phone_number = "+1234567890"  # Replace with actual phone number
        human_count = 3  # Replace with actual detection count
        
        # Send alert
        if sms.send_detection_alert(phone_number, human_count):
            print("Alert sent successfully")
        else:
            print("Failed to send alert")
            
        # Disconnect
        sms.disconnect()
