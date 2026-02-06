#!/usr/bin/env python3
# coding: utf-8
"""
Automotive AGI Dashboard Assistant
An intelligent vehicle assistant that monitors OBD-II data, analyzes road conditions via camera,
provides navigation, and converses with the driver for enhanced safety and convenience.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter, deque
from datetime import datetime, timedelta
import json
import threading
from queue import Queue, Empty
from enum import Enum

# Load environment variables from .env file
from dotenv import load_dotenv

import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr
from openai import OpenAI
from gtts import gTTS
import requests
from geopy.geocoders import Nominatim
from ultralytics import YOLO
import cv2
import numpy as np

# OBD2 imports
import obdii

load_dotenv()


# ============================================================================
# Enums and Constants
# ============================================================================

class VehicleState(Enum):
    """Vehicle operational states."""
    IDLE = "idle"
    DRIVING = "driving"
    PARKED = "parked"
    WARNING = "warning"
    CRITICAL = "critical"



# ============================================================================
# PyTorch 2.6+ Compatibility: Fix weights_only loading issue
# ============================================================================
try:
    import torch
    # Monkey-patch torch.load to use weights_only=False for compatibility
    _original_torch_load = torch.load
    def patched_torch_load(f, *args, **kwargs):
        # Force weights_only=False for older model formats
        kwargs.setdefault('weights_only', False)
        return _original_torch_load(f, *args, **kwargs)
    torch.load = patched_torch_load
except Exception as e:
    pass  # Patching failed, will attempt to load models anyway


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


# ============================================================================
# Configuration Management
# ============================================================================

@dataclass
class DashboardConfig:
    """Comprehensive configuration for dashboard AGI system."""
    
    # API Keys (loaded from environment variables for security)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Audio Settings
    SAMPLE_RATE: int = 44100
    RECORDING_DURATION: int = 5
    AUDIO_CHANNELS: int = 2
    AUDIO_DTYPE: str = 'int16'
    VOICE_ACTIVATION_THRESHOLD: float = 500.0
    
    # Vision Settings
    CAMERA_INDEX: int = 0
    VISION_ANALYSIS_INTERVAL: int = 180  # 3 minutes in seconds
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720
    SAVE_ANALYZED_FRAMES: bool = True
    
    # OBD-II Settings
    OBD_PORT: str = "/dev/ttyUSB0"  # Serial port for OBD-II adapter
    OBD_BAUDRATE: int = 38400
    OBD_POLL_INTERVAL: int = 2  # seconds
    OBD_ENABLED: bool = True
    
    # File Paths
    OUTPUT_DIR: Path = Path("dashboard_data")
    AUDIO_FILE: str = "driver_voice.wav"
    SPEECH_FILE: str = "assistant_response.mp3"
    LOG_FILE: str = "dashboard_agi.log"
    FRAME_DIR: str = "camera_frames"
    OBD_LOG: str = "obd_data.jsonl"
    
    # LLM Settings
    LLM_MODEL: str = "gpt-4-turbo-preview"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 800
    LLM_SYSTEM_ROLE: str = (
        "You are an advanced automotive AGI assistant integrated into a vehicle's dashboard system. "
        "You have access to real-time OBD-II diagnostics, camera vision analysis, GPS navigation, "
        "and conversational AI capabilities. Your primary goals are: "
        "1) Ensure driver and passenger safety through proactive monitoring and alerts "
        "2) Provide helpful navigation and route optimization "
        "3) Diagnose vehicle issues and recommend maintenance "
        "4) Engage in natural, context-aware conversation "
        "5) Maintain situational awareness of traffic, weather, and road conditions. "
        "Be concise, professional, and prioritize safety above all else. "
        "Speak naturally as if you're a knowledgeable co-pilot."
    )
    YOLO_MODEL: str = "yolov8n.pt"
    YOLO_CONFIDENCE: float = 0.5
    YOLO_IOU_THRESHOLD: float = 0.45
    
    # Geolocation Settings
    GEO_USER_AGENT: str = "automotive_agi_assistant/2.0"
    GPS_UPDATE_INTERVAL: int = 30  # seconds
    
    # Operational Settings
    MAIN_LOOP_DELAY: float = 0.5
    MAX_RETRIES: int = 3
    REQUEST_TIMEOUT: int = 10
    CONTEXT_MEMORY_SIZE: int = 50  # Number of recent interactions to remember
    
    # Safety Thresholds
    MAX_ENGINE_TEMP: float = 105.0  # Celsius
    MIN_BATTERY_VOLTAGE: float = 11.5  # Volts
    MAX_RPM_WARNING: int = 6000
    LOW_FUEL_THRESHOLD: float = 15.0  # Percent
    
    # Alert Settings
    ENABLE_VOICE_ALERTS: bool = True
    ALERT_COOLDOWN: int = 60  # seconds between repeat alerts
    
    def __post_init__(self):
        """Validate configuration and create necessary directories."""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (self.OUTPUT_DIR / self.FRAME_DIR).mkdir(parents=True, exist_ok=True)
        
        if not self.OPENAI_API_KEY:
            logging.warning("OPENAI_API_KEY not set in environment variables")
    
    @classmethod
    def from_file(cls, config_path: str) -> 'DashboardConfig':
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return cls(**config_data)
        except FileNotFoundError:
            logging.warning(f"Config file {config_path} not found, using defaults")
            return cls()
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in config file: {e}")
            return cls()


# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging(config: DashboardConfig) -> logging.Logger:
    """Configure comprehensive logging with file and console handlers."""
    
    logger = logging.getLogger("DashboardAGI")
    logger.setLevel(logging.DEBUG)
    
    if logger.handlers:
        logger.handlers.clear()
    
    # File handler - detailed logs
    file_handler = logging.FileHandler(
        config.OUTPUT_DIR / config.LOG_FILE,
        mode='a',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler - important logs only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ============================================================================
# Exception Classes
# ============================================================================

class DashboardAGIError(Exception):
    """Base exception for dashboard AGI errors."""
    pass


class OBDConnectionError(DashboardAGIError):
    """Error connecting to OBD-II interface."""
    pass


class CameraError(DashboardAGIError):
    """Error with camera system."""
    pass


class SafetyCriticalError(DashboardAGIError):
    """Critical safety issue detected."""
    pass


# ============================================================================
# OBD-II Module
# ============================================================================

@dataclass
class OBDData:
    """Container for OBD-II diagnostic data."""
    timestamp: datetime
    engine_rpm: Optional[float] = None
    vehicle_speed: Optional[float] = None  # km/h
    engine_temp: Optional[float] = None  # Celsius
    throttle_position: Optional[float] = None  # Percent
    fuel_level: Optional[float] = None  # Percent
    battery_voltage: Optional[float] = None  # Volts
    intake_air_temp: Optional[float] = None  # Celsius
    maf_rate: Optional[float] = None  # Mass air flow (g/s)
    engine_load: Optional[float] = None  # Percent
    dtc_codes: List[str] = field(default_factory=list)  # Diagnostic trouble codes
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'engine_rpm': self.engine_rpm,
            'vehicle_speed': self.vehicle_speed,
            'engine_temp': self.engine_temp,
            'throttle_position': self.throttle_position,
            'fuel_level': self.fuel_level,
            'battery_voltage': self.battery_voltage,
            'intake_air_temp': self.intake_air_temp,
            'maf_rate': self.maf_rate,
            'engine_load': self.engine_load,
            'dtc_codes': self.dtc_codes,
            'raw_data': self.raw_data
        }


class OBDInterface:
    """
    Interface for OBD-II vehicle diagnostics.
    Monitors engine parameters, fuel levels, DTCs, and vehicle health.
    """
    
    def __init__(self, config: DashboardConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.connection = None
        self.is_connected = False
        self.last_data: Optional[OBDData] = None
        self.data_history: deque = deque(maxlen=1000)
        
        if config.OBD_ENABLED:
            self._initialize_connection()
    
    def _initialize_connection(self):
        """
        Initialize connection to OBD-II adapter.
        This is a placeholder - actual implementation would use python-OBD or similar library.
        """
        try:
            self.logger.info(f"Initializing OBD-II connection on {self.config.OBD_PORT}")
            
            # TODO: Implement actual OBD-II connection
            # import obd
            # self.connection = obd.OBD(portstr=self.config.OBD_PORT, baudrate=self.config.OBD_BAUDRATE)
            # self.is_connected = self.connection.is_connected()

            # OBD2 implementation
            conn = obdii.Connection(("127.0.0.1", 35000))

            if conn.is_connected():
                self.connection = conn
                self.is_connected = True
                self.logger.info("OBD-II connection established successfully")
            
            # Placeholder for development
            #self.is_connected = False
            #self.logger.warning("OBD-II interface not implemented - using simulated data")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to OBD-II: {e}")
            self.is_connected = False
    
    def read_data(self) -> OBDData:
        """
        Read current OBD-II data from vehicle.
        
        Returns:
            OBDData object with current diagnostics
        """
        try:
            if not self.is_connected:
                # Return simulated data for development
                return self._get_simulated_data()
            
            # TODO: Implement actual OBD-II data reading
            # data = OBDData(timestamp=datetime.now())
            # data.engine_rpm = self.connection.query(obd.commands.RPM).value
            # data.vehicle_speed = self.connection.query(obd.commands.SPEED).value
            # data.engine_temp = self.connection.query(obd.commands.COOLANT_TEMP).value
            # ... etc

            data = OBDData(timestamp=datetime.now())

            try:
                data.engine_rpm = self.connection.query(obdii.commands.ENGINE_SPEED).value
                data.vehicle_speed = self.connection.query(obdii.commands.VEHICLE_SPEED).value
                #data.engine_temp = self.connection.query(commands.ENGINE_COOLANT_TEMP).value
                #data.throttle_position = self.connection.query(commands.THROTTLE_POSITION).value
                #data.fuel_level = self.connection.query(commands.FUEL_LEVEL).value

            except Exception as e:
                self.logger.warning("Some OBD-II data points are missing")
            
            return OBDData(timestamp=datetime.now())
            
        except Exception as e:
            self.logger.error(f"Error reading OBD data: {e}")
            return OBDData(timestamp=datetime.now())
    
    def _get_simulated_data(self) -> OBDData:
        """Generate simulated OBD data for testing."""
        import random
        
        data = OBDData(
            timestamp=datetime.now(),
            engine_rpm=random.uniform(800, 3500),
            vehicle_speed=random.uniform(0, 100),
            engine_temp=random.uniform(85, 95),
            throttle_position=random.uniform(10, 60),
            fuel_level=random.uniform(20, 80),
            battery_voltage=random.uniform(12.5, 14.5),
            intake_air_temp=random.uniform(20, 40),
            engine_load=random.uniform(20, 70),
            dtc_codes=[]
        )
        
        self.last_data = data
        self.data_history.append(data)
        return data
    
    def check_health(self, data: OBDData) -> List[Tuple[AlertLevel, str]]:
        """
        Analyze OBD data for health issues.
        
        Args:
            data: OBD data to analyze
            
        Returns:
            List of (alert_level, message) tuples
        """
        alerts = []
        
        # Engine temperature check
        if data.engine_temp and data.engine_temp > self.config.MAX_ENGINE_TEMP:
            alerts.append((
                AlertLevel.CRITICAL,
                f"Engine temperature critically high: {data.engine_temp:.1f}°C"
            ))
        elif data.engine_temp and data.engine_temp > self.config.MAX_ENGINE_TEMP - 10:
            alerts.append((
                AlertLevel.WARNING,
                f"Engine temperature elevated: {data.engine_temp:.1f}°C"
            ))
        
        # Battery voltage check
        if data.battery_voltage and data.battery_voltage < self.config.MIN_BATTERY_VOLTAGE:
            alerts.append((
                AlertLevel.WARNING,
                f"Battery voltage low: {data.battery_voltage:.1f}V"
            ))
        
        # RPM check
        if data.engine_rpm and data.engine_rpm > self.config.MAX_RPM_WARNING:
            alerts.append((
                AlertLevel.WARNING,
                f"Engine RPM very high: {data.engine_rpm:.0f}"
            ))
        
        # Fuel level check
        if data.fuel_level and data.fuel_level < self.config.LOW_FUEL_THRESHOLD:
            alerts.append((
                AlertLevel.INFO,
                f"Fuel level low: {data.fuel_level:.1f}%"
            ))
        
        # DTC codes check
        if data.dtc_codes:
            alerts.append((
                AlertLevel.WARNING,
                f"Diagnostic trouble codes detected: {', '.join(data.dtc_codes)}"
            ))
        
        return alerts
    
    def log_data(self, data: OBDData):
        """Log OBD data to file."""
        try:
            log_path = self.config.OUTPUT_DIR / self.config.OBD_LOG
            with open(log_path, 'a') as f:
                f.write(json.dumps(data.to_dict()) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to log OBD data: {e}")
    
    def get_summary(self) -> str:
        """Get human-readable summary of current vehicle state."""
        if not self.last_data:
            return "No vehicle data available"
        
        d = self.last_data
        return (
            f"Speed: {d.vehicle_speed:.0f} km/h, "
            f"RPM: {d.engine_rpm:.0f}, "
            f"Temp: {d.engine_temp:.1f}°C, "
            f"Fuel: {d.fuel_level:.0f}%"
        )


# ============================================================================
# Vision/Camera Module
# ============================================================================

@dataclass
class VisionAnalysis:
    """Container for camera vision analysis results."""
    timestamp: datetime
    detected_objects: Dict[str, int]  # object_name: count
    scene_description: str
    hazards_detected: List[str]
    traffic_density: str  # "light", "moderate", "heavy"
    weather_conditions: str  # "clear", "rainy", "foggy", etc.
    road_conditions: str  # "good", "poor", "hazardous"
    confidence_score: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'detected_objects': self.detected_objects,
            'scene_description': self.scene_description,
            'hazards_detected': self.hazards_detected,
            'traffic_density': self.traffic_density,
            'weather_conditions': self.weather_conditions,
            'road_conditions': self.road_conditions,
            'confidence_score': self.confidence_score
        }


class DashcamVision:
    """
    Computer vision system for real-time road monitoring.
    Analyzes dashcam feed for objects, hazards, traffic, and road conditions.
    """
    
    def __init__(self, config: DashboardConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.camera = None
        self.model = None
        self.last_analysis: Optional[VisionAnalysis] = None
        self.analysis_history: deque = deque(maxlen=100)
        
        self._initialize_camera()
        self._load_model()
    
    def _initialize_camera(self):
        """Initialize camera connection."""
        try:
            self.logger.info(f"Initializing camera {self.config.CAMERA_INDEX}")
            self.camera = cv2.VideoCapture(self.config.CAMERA_INDEX)
            
            if self.camera.isOpened():
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.FRAME_WIDTH)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.FRAME_HEIGHT)
                self.logger.info("Camera initialized successfully")
            else:
                self.logger.warning("Camera not available - using simulated vision")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            self.camera = None
    
    def _load_model(self):
        """Load YOLO model for object detection."""
        try:
            self.logger.info(f"Loading YOLO model: {self.config.YOLO_MODEL}")
            self.model = YOLO(self.config.YOLO_MODEL)
            self.logger.info("YOLO model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the dashcam.
        
        Returns:
            Numpy array of frame or None if failed
        """
        if not self.camera or not self.camera.isOpened():
            self.logger.warning("Camera not available")
            return None
        
        try:
            ret, frame = self.camera.read()
            if ret:
                return frame
            else:
                self.logger.warning("Failed to capture frame")
                return None
        except Exception as e:
            self.logger.error(f"Error capturing frame: {e}")
            return None
    
    def analyze_frame(self, frame: np.ndarray) -> VisionAnalysis:
        """
        Perform comprehensive analysis of captured frame.
        
        Args:
            frame: Image frame to analyze
            
        Returns:
            VisionAnalysis object with detected objects and conditions
        """

        try:
            # Run YOLO detection
            results = self.model(
                frame,
                conf=self.config.YOLO_CONFIDENCE,
                iou=self.config.YOLO_IOU_THRESHOLD,
                verbose=False
            )
            
            # Extract detections
            detections = results[0].boxes
            class_ids = detections.cls.cpu().numpy() if len(detections) > 0 else []
            names = results[0].names
            
            # Count objects
            detected_objects = {}
            for class_id in class_ids:
                obj_name = names[int(class_id)]
                detected_objects[obj_name] = detected_objects.get(obj_name, 0) + 1
            
            # Analyze scene
            hazards = self._detect_hazards(detected_objects)
            traffic = self._assess_traffic_density(detected_objects)
            weather = self._assess_weather(frame)
            road = self._assess_road_conditions(frame, detected_objects)
            
            # Generate description
            description = self._generate_scene_description(detected_objects, hazards)
            
            analysis = VisionAnalysis(
                timestamp=datetime.now(),
                detected_objects=detected_objects,
                scene_description=description,
                hazards_detected=hazards,
                traffic_density=traffic,
                weather_conditions=weather,
                road_conditions=road,
                confidence_score=0.85  # Placeholder
            )
            
            self.last_analysis = analysis
            self.analysis_history.append(analysis)
            
            self.logger.info(f"Vision analysis: {len(detected_objects)} object types detected")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Frame analysis failed: {e}")
            return VisionAnalysis(
                timestamp=datetime.now(),
                detected_objects={},
                scene_description="Analysis failed",
                hazards_detected=[],
                traffic_density="unknown",
                weather_conditions="unknown",
                road_conditions="unknown",
                confidence_score=0.0
            )
    
    def _detect_hazards(self, objects: Dict[str, int]) -> List[str]:
        """Identify potential hazards from detected objects."""
        hazards = []
        
        # Check for pedestrians
        if objects.get('person', 0) > 0:
            hazards.append(f"Pedestrians detected ({objects['person']})")
        
        # Check for cyclists
        if objects.get('bicycle', 0) > 0:
            hazards.append(f"Cyclists detected ({objects['bicycle']})")
        
        # Check for motorcycles
        if objects.get('motorcycle', 0) > 0:
            hazards.append(f"Motorcycles nearby ({objects['motorcycle']})")
        
        # Check for animals
        for animal in ['dog', 'cat', 'bird', 'horse']:
            if objects.get(animal, 0) > 0:
                hazards.append(f"Animal on road ({animal})")
        
        # Check for trucks
        if objects.get('truck', 0) > 0:
            hazards.append("Large vehicles in vicinity")
        
        return hazards
    
    def _assess_traffic_density(self, objects: Dict[str, int]) -> str:
        """Assess traffic density from vehicle count."""
        vehicle_types = ['car', 'truck', 'bus', 'motorcycle']
        total_vehicles = sum(objects.get(v, 0) for v in vehicle_types)
        
        if total_vehicles == 0:
            return "clear"
        elif total_vehicles <= 3:
            return "light"
        elif total_vehicles <= 8:
            return "moderate"
        else:
            return "heavy"
    
    def _assess_weather(self, frame: np.ndarray) -> str:
        """
        Assess weather conditions from image brightness and characteristics.
        This is a simplified placeholder - real implementation would use more sophisticated CV.
        """
        # Calculate average brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        if avg_brightness < 50:
            return "dark/night"
        elif avg_brightness < 100:
            return "overcast"
        else:
            return "clear"
    
    def _assess_road_conditions(self, frame: np.ndarray, objects: Dict) -> str:
        """
        Assess road conditions.
        Placeholder for more sophisticated analysis.
        """
        # In real implementation, would analyze road surface, lane markings, etc.
        return "good"
    
    def _generate_scene_description(self, objects: Dict, hazards: List[str]) -> str:
        """Generate natural language description of scene."""
        if not objects:
            return "Road ahead appears clear"
        
        desc_parts = []
        
        # Describe vehicles
        vehicles = []
        for v_type in ['car', 'truck', 'bus', 'motorcycle']:
            count = objects.get(v_type, 0)
            if count > 0:
                vehicles.append(f"{count} {v_type}{'s' if count > 1 else ''}")
        
        if vehicles:
            desc_parts.append(f"Traffic: {', '.join(vehicles)}")
        
        # Describe hazards
        if hazards:
            desc_parts.append(f"Alerts: {len(hazards)} potential hazards")
        
        return "; ".join(desc_parts) if desc_parts else "Normal traffic conditions"
    
    def save_frame(self, frame: np.ndarray, prefix: str = "frame"):
        """Save frame to disk with timestamp."""
        if not self.config.SAVE_ANALYZED_FRAMES:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.jpg"
            filepath = self.config.OUTPUT_DIR / self.config.FRAME_DIR / filename
            cv2.imwrite(str(filepath), frame)
            self.logger.debug(f"Frame saved: {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save frame: {e}")
    
    def cleanup(self):
        """Release camera resources."""
        if self.camera:
            self.camera.release()
            self.logger.info("Camera released")


# ============================================================================
# Navigation Module
# ============================================================================

class NavigationService:
    """Handle GPS location, routing, and navigation."""
    
    def __init__(self, config: DashboardConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.geolocator = Nominatim(user_agent=config.GEO_USER_AGENT)
        self.current_location: Optional[Dict] = None
        self.current_route: Optional[Dict] = None
        self.destination: Optional[Tuple[float, float]] = None
    
    def get_current_location(self) -> Dict[str, str]:
        """Get current location based on IP address."""
        try:
            self.logger.info("Fetching current location via IP")
            response = requests.get(
                "https://ipinfo.io/json",
                timeout=self.config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            
            location_info = {
                "city": data.get("city", "Unknown"),
                "region": data.get("region", "Unknown"),
                "country": data.get("country", "Unknown"),
                "coordinates": data.get("loc", "Unknown")
            }
            
            self.current_location = location_info
            self.logger.info(f"Location: {location_info['city']}, {location_info['country']}")
            return location_info
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to get location: {e}")
            return {"error": str(e)}
    
    def geocode_address(self, address: str) -> Optional[Tuple[float, float]]:
        """Convert address to coordinates."""
        try:
            self.logger.info(f"Geocoding address: {address}")
            location = self.geolocator.geocode(address)
            
            if location:
                coords = (location.longitude, location.latitude)
                self.logger.info(f"Geocoded to: {coords}")
                return coords
            else:
                self.logger.warning(f"Address not found: {address}")
                return None
                
        except Exception as e:
            self.logger.error(f"Geocoding failed: {e}")
            return None
    
    def set_destination(self, destination: str) -> bool:
        """Set navigation destination."""
        coords = self.geocode_address(destination)
        if coords:
            self.destination = coords
            self.logger.info(f"Destination set: {destination}")
            return True
        return False
    
    def get_directions(
        self,
        start_coords: Tuple[float, float],
        end_coords: Tuple[float, float]
    ) -> Dict[str, any]:
        """Get driving directions between two points."""
        try:
            start_lon, start_lat = start_coords
            end_lon, end_lat = end_coords
            
            url = (
                f"http://router.project-osrm.org/route/v1/driving/"
                f"{start_lon},{start_lat};{end_lon},{end_lat}"
                f"?overview=full&geometries=geojson&steps=true"
            )
            
            self.logger.info("Requesting route from OSRM")
            
            response = requests.get(url, timeout=self.config.REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            if "routes" not in data or not data["routes"]:
                self.logger.error("No routes found")
                return {"error": "No routes found"}
            
            route = data["routes"][0]
            leg = route["legs"][0]
            
            # Extract turn-by-turn directions
            directions = []
            for i, step in enumerate(leg["steps"], 1):
                instruction = step["maneuver"]["type"]
                road = step.get("name", "unnamed road")
                distance = step["distance"]
                
                directions.append({
                    "step": i,
                    "instruction": instruction,
                    "road": road,
                    "distance_m": distance
                })
            
            route_info = {
                "distance_km": round(route["distance"] / 1000, 2),
                "duration_min": round(route["duration"] / 60, 2),
                "directions": directions
            }
            
            self.current_route = route_info
            self.logger.info(f"Route calculated: {route_info['distance_km']} km")
            
            return route_info
            
        except requests.RequestException as e:
            self.logger.error(f"Routing request failed: {e}")
            return {"error": str(e)}


# ============================================================================
# Audio Module
# ============================================================================

class AudioHandler:
    """Handle audio recording, transcription, and speech synthesis."""
    
    def __init__(self, config: DashboardConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.recognizer = sr.Recognizer()
        self.is_listening = False
    
    def record_audio(self, duration: Optional[int] = None) -> str:
        """Record audio from microphone."""
        try:
            duration = duration or self.config.RECORDING_DURATION
            self.logger.info(f"Recording audio for {duration} seconds...")
            
            recorded_audio = sd.rec(
                int(duration * self.config.SAMPLE_RATE),
                samplerate=self.config.SAMPLE_RATE,
                channels=self.config.AUDIO_CHANNELS,
                dtype=self.config.AUDIO_DTYPE
            )
            sd.wait()
            
            audio_path = self.config.OUTPUT_DIR / self.config.AUDIO_FILE
            write(str(audio_path), self.config.SAMPLE_RATE, recorded_audio)
            
            self.logger.info(f"Audio saved to: {audio_path}")
            return str(audio_path)
            
        except Exception as e:
            self.logger.error(f"Audio recording failed: {e}")
            raise
    
    def transcribe_audio(self, audio_file: str) -> str:
        """Transcribe audio file to text."""
        try:
            if not os.path.exists(audio_file):
                raise FileNotFoundError(f"Audio file not found: {audio_file}")
            
            self.logger.info(f"Transcribing audio: {audio_file}")
            
            with sr.AudioFile(audio_file) as source:
                audio = self.recognizer.record(source)
            
            text = self.recognizer.recognize_google(audio)
            self.logger.info(f"Transcription: '{text}'")
            return text
            
        except sr.UnknownValueError:
            self.logger.warning("Could not understand audio")
            return ""
        except sr.RequestError as e:
            self.logger.error(f"Transcription service error: {e}")
            return ""
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            return ""
    
    def speak_text(self, text: str, priority: bool = False) -> bool:
        """Convert text to speech and play it with unique filename to avoid file lock issues.
        
        CHANGE: Fixed 'Permission denied' error by using unique filenames instead of overwriting
        the same file. Windows Media Player was locking the file, preventing overwrites.
        Solution: Generate unique filename with timestamp for each speech call.
        """
        try:
            if not text:
                return False
            
            self.logger.info(f"Speaking: '{text[:50]}...'")  
            
            # CHANGE: Generate speech with unique filename using timestamp
            # Format: assistant_20260206_123045_123456.mp3
            # This ensures no file lock conflicts between consecutive speak_text calls
            speech = gTTS(text=text, lang='en', slow=False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:18]
            unique_speech_file = f"assistant_{timestamp}.mp3"
            speech_path = self.config.OUTPUT_DIR / unique_speech_file
            speech.save(str(speech_path))
            
            # CHANGE: Play audio (platform-specific)
            # Each file has unique name, so Windows Media Player won't lock previous file
            if os.name == 'posix':  # Linux/Unix
                os.system(f"mpg123 -q {speech_path} &")
            elif os.name == 'nt':  # Windows
                os.system(f"start {speech_path}")
            
            # CHANGE: Periodically cleanup old speech files to prevent disk space issues
            # Keep only the 10 most recent audio files
            if timestamp.endswith('0'):  # Cleanup every ~10 calls to reduce overhead
                self.cleanup_old_speech_files(keep_recent=10)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Text-to-speech failed: {e}")
            return False
    
    def cleanup_old_speech_files(self, keep_recent: int = 10) -> None:
        """Remove old speech files to prevent disk space accumulation.
        
        CHANGE: New method to manage cleanup of generated speech files.
        Without cleanup, thousands of MP3 files would accumulate over time.
        Keeps only the N most recent files based on modification time.
        
        Args:
            keep_recent: Number of recent files to keep (default: 10)
        """
        try:
            # Get the speech output directory
            speech_dir = self.config.OUTPUT_DIR
            
            # CHANGE: Find all assistant_*.mp3 files sorted by modification time (newest first)
            speech_files = sorted(
                speech_dir.glob("assistant_*.mp3"),
                key=lambda f: f.stat().st_mtime,
                reverse=True  # Newest first
            )
            
            # CHANGE: Delete files older than the keep_recent threshold
            # This safely removes old MP3s without affecting current playback
            for old_file in speech_files[keep_recent:]:
                try:
                    old_file.unlink()  # Delete the file
                    self.logger.debug(f"Cleaned up old speech file: {old_file.name}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete {old_file.name}: {e}")
        except Exception as e:
            self.logger.warning(f"Cleanup of speech files failed: {e}")

# ============================================================================
# AGI Brain Module
# ============================================================================

class AGIBrain:
    """
    Central AGI intelligence that processes all inputs and generates responses.
    Integrates OBD data, vision analysis, navigation, and conversation.
    """
    
    def __init__(self, config: DashboardConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.client = None
        self.conversation_history: deque = deque(maxlen=config.CONTEXT_MEMORY_SIZE)
        self.last_alert_time: Dict[str, datetime] = {}
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        if not self.config.OPENAI_API_KEY:
            self.logger.error("OpenAI API key not configured")
            return
        
        try:
            import httpx
            # Use a simple httpx client to avoid proxy compatibility issues
            http_client = httpx.Client()
            self.client = OpenAI(api_key=self.config.OPENAI_API_KEY, http_client=http_client)
            self.logger.info("OpenAI client initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
    
    def build_context(
        self,
        obd_data: Optional[OBDData] = None,
        vision_analysis: Optional[VisionAnalysis] = None,
        navigation_info: Optional[Dict] = None
    ) -> str:
        """Build comprehensive context from all systems."""
        context_parts = ["=== VEHICLE SYSTEM STATUS ==="]
        
        # OBD Data
        if obd_data:
            context_parts.append("\n[Vehicle Diagnostics]")
            context_parts.append(f"Speed: {obd_data.vehicle_speed:.0f} km/h")
            context_parts.append(f"Engine RPM: {obd_data.engine_rpm:.0f}")
            context_parts.append(f"Engine Temperature: {obd_data.engine_temp:.1f}°C")
            context_parts.append(f"Throttle Position: {obd_data.throttle_position:.1f}%")
            context_parts.append(f"Fuel Level: {obd_data.fuel_level:.1f}%")
            context_parts.append(f"Battery Voltage: {obd_data.battery_voltage:.1f}V")
            if obd_data.dtc_codes:
                context_parts.append(f"Trouble Codes: {', '.join(obd_data.dtc_codes)}")
        
        # Vision Analysis
        if vision_analysis:
            context_parts.append("\n[Road Conditions]")
            context_parts.append(f"Scene: {vision_analysis.scene_description}")
            context_parts.append(f"Traffic Density: {vision_analysis.traffic_density}")
            context_parts.append(f"Weather: {vision_analysis.weather_conditions}")
            
            if vision_analysis.detected_objects:
                obj_list = [f"{count} {name}" for name, count in vision_analysis.detected_objects.items()]
                context_parts.append(f"Detected: {', '.join(obj_list)}")
            
            if vision_analysis.hazards_detected:
                context_parts.append(f"⚠️ Hazards: {'; '.join(vision_analysis.hazards_detected)}")
        
        # Navigation
        if navigation_info:
            context_parts.append("\n[Navigation]")
            # FIX: Check if current_location exists AND is not None before calling .get()
            # Previous bug: checked if key existed but didn't validate the value was a dict
            if navigation_info.get('current_location') is not None:
                loc = navigation_info['current_location']
                context_parts.append(f"Location: {loc.get('city', 'Unknown')}")
            # FIX: Check if route exists AND is not None before calling .get()
            # This prevents 'NoneType' object has no attribute 'get' error
            if navigation_info.get('route') is not None:
                route = navigation_info['route']
                context_parts.append(f"Route: {route.get('distance_km')} km, {route.get('duration_min')} min")
        
        return "\n".join(context_parts)
    
    def process_input(
        self,
        user_input: str,
        obd_data: Optional[OBDData] = None,
        vision_analysis: Optional[VisionAnalysis] = None,
        navigation_info: Optional[Dict] = None
    ) -> str:
        """
        Process user input with full system context.
        
        Args:
            user_input: Driver's spoken command or query
            obd_data: Current vehicle diagnostics
            vision_analysis: Current road/vision analysis
            navigation_info: Current navigation state
            
        Returns:
            AGI's response text
        """
        if not self.client:
            return "I'm not fully initialized yet. Please check the system configuration."
        
        try:
            # Build system context
            context = self.build_context(obd_data, vision_analysis, navigation_info)
            
            # Build conversation history
            messages = [
                {"role": "system", "content": self.config.LLM_SYSTEM_ROLE},
                {"role": "system", "content": context}
            ]
            
            # Add conversation history
            for msg in list(self.conversation_history)[-10:]:  # Last 10 exchanges
                messages.append(msg)
            
            # Add current user input
            messages.append({"role": "user", "content": user_input})
            
            self.logger.info(f"Processing: '{user_input[:50]}...'")
            
            # Call LLM
            response = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=messages,
                temperature=self.config.LLM_TEMPERATURE,
                max_tokens=self.config.LLM_MAX_TOKENS
            )
            
            response_text = response.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            self.logger.info(f"AGI response: '{response_text[:50]}...'")
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"AGI processing failed: {e}")
            return "I encountered an error processing that. Could you repeat?"
    
    def generate_proactive_alert(
        self,
        alert_type: str,
        alert_level: AlertLevel,
        message: str
    ) -> Optional[str]:
        """
        Generate proactive alert if needed.
        Uses cooldown to avoid alert spam.
        
        Returns:
            Alert message to speak, or None if in cooldown
        """
        now = datetime.now()
        last_alert = self.last_alert_time.get(alert_type)
        
        if last_alert:
            time_since = (now - last_alert).total_seconds()
            if time_since < self.config.ALERT_COOLDOWN:
                return None
        
        self.last_alert_time[alert_type] = now
        
        # Generate contextual alert message
        if alert_level == AlertLevel.CRITICAL:
            prefix = "⚠️ CRITICAL ALERT: "
        elif alert_level == AlertLevel.WARNING:
            prefix = "Warning: "
        else:
            prefix = "Notice: "
        
        return prefix + message


# ============================================================================
# Main Dashboard AGI System
# ============================================================================

class DashboardAGI:
    """
    Main orchestrator for the automotive AGI dashboard system.
    Coordinates all subsystems and manages the main event loop.
    """
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.logger = setup_logging(config)
        
        self.logger.info("=" * 80)
        self.logger.info("DASHBOARD AGI SYSTEM INITIALIZING")
        self.logger.info("=" * 80)
        
        # Initialize subsystems
        self.obd = OBDInterface(config, self.logger)
        self.vision = DashcamVision(config, self.logger)
        self.navigation = NavigationService(config, self.logger)
        self.audio = AudioHandler(config, self.logger)
        self.brain = AGIBrain(config, self.logger)
        
        # State management
        self.running = False
        self.vehicle_state = VehicleState.IDLE
        self.last_vision_analysis = datetime.now()
        self.last_obd_read = datetime.now()
        
        # Threading
        self.obd_thread: Optional[threading.Thread] = None
        self.vision_thread: Optional[threading.Thread] = None
        self.alert_queue: Queue = Queue()
        
        self.logger.info("Dashboard AGI initialized successfully")
    
    def start_background_tasks(self):
        """Start background monitoring threads."""
        self.logger.info("Starting background monitoring tasks")
        
        # OBD monitoring thread
        self.obd_thread = threading.Thread(target=self._obd_monitor_loop, daemon=True)
        self.obd_thread.start()
        
        # Vision monitoring thread
        self.vision_thread = threading.Thread(target=self._vision_monitor_loop, daemon=True)
        self.vision_thread.start()
    
    def _obd_monitor_loop(self):
        """Background thread for continuous OBD monitoring."""
        self.logger.info("OBD monitor thread started")
        
        while self.running:
            try:
                # Read OBD data
                obd_data = self.obd.read_data()
                self.last_obd_read = datetime.now()
                
                # Log data
                self.obd.log_data(obd_data)
                
                # Check for health issues
                alerts = self.obd.check_health(obd_data)
                for alert_level, message in alerts:
                    self.alert_queue.put((alert_level, message))
                
                # Sleep until next poll
                time.sleep(self.config.OBD_POLL_INTERVAL)
                
            except Exception as e:
                self.logger.error(f"OBD monitor error: {e}")
                time.sleep(5)
    
    def _vision_monitor_loop(self):
        """Background thread for periodic vision analysis."""
        self.logger.info("Vision monitor thread started")
        
        while self.running:
            try:
                # Capture and analyze frame
                frame = self.vision.capture_frame()
                if frame is not None:
                    analysis = self.vision.analyze_frame(frame)
                    self.last_vision_analysis = datetime.now()
                    
                    # Save frame
                    self.vision.save_frame(frame)
                    
                    # Check for hazards
                    if analysis.hazards_detected:
                        for hazard in analysis.hazards_detected:
                            self.alert_queue.put((AlertLevel.WARNING, hazard))
                
                # Sleep until next analysis
                time.sleep(self.config.VISION_ANALYSIS_INTERVAL)
                
            except Exception as e:
                self.logger.error(f"Vision monitor error: {e}")
                time.sleep(30)
    
    def process_alerts(self):
        """Process any pending alerts from background threads."""
        try:
            while not self.alert_queue.empty():
                alert_level, message = self.alert_queue.get_nowait()
                
                # Generate alert response
                alert_msg = self.brain.generate_proactive_alert(
                    alert_type=message[:20],  # Use first 20 chars as type
                    alert_level=alert_level,
                    message=message
                )
                
                if alert_msg and self.config.ENABLE_VOICE_ALERTS:
                    self.logger.warning(f"ALERT: {alert_msg}")
                    self.audio.speak_text(alert_msg, priority=True)
                    
        except Empty:
            pass
        except Exception as e:
            self.logger.error(f"Error processing alerts: {e}")
    
    def handle_voice_interaction(self):
        """Handle one voice interaction cycle."""
        try:
            # Record audio
            self.logger.info("Listening for driver input...")
            audio_file = self.audio.record_audio()
            
            # Transcribe
            user_input = self.audio.transcribe_audio(audio_file)
            
            if not user_input:
                return
            
            self.logger.info(f"Driver: '{user_input}'")
            
            # Check for exit command
            if any(word in user_input.lower() for word in ["shutdown", "exit", "quit"]):
                self.audio.speak_text("Shutting down dashboard AGI. Drive safely.")
                self.running = False
                return
            
            # Get current system state
            obd_data = self.obd.last_data
            vision_analysis = self.vision.last_analysis
            navigation_info = {
                'current_location': self.navigation.current_location,
                'route': self.navigation.current_route
            }
            
            # Process with AGI brain
            response = self.brain.process_input(
                user_input,
                obd_data,
                vision_analysis,
                navigation_info
            )
            
            # Speak response
            self.audio.speak_text(response)
            
        except Exception as e:
            self.logger.error(f"Voice interaction error: {e}")
            self.audio.speak_text("I didn't catch that. Could you repeat?")
    
    def run(self):
        """Main event loop."""
        self.running = True
        self.logger.info("=" * 80)
        self.logger.info("DASHBOARD AGI IS NOW ACTIVE")
        self.logger.info("=" * 80)
        
        try:
            # Start background monitoring
            self.start_background_tasks()
            
            # Initial greeting
            self.audio.speak_text(
                "Dashboard AGI online. All systems operational. How can I assist you?"
            )
            
            # Main loop
            while self.running:
                # Process any alerts
                self.process_alerts()
                
                # Handle voice interaction
                self.handle_voice_interaction()
                
                # Small delay
                time.sleep(self.config.MAIN_LOOP_DELAY)
                
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Critical error in main loop: {e}", exc_info=True)
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown of all systems."""
        self.logger.info("=" * 80)
        self.logger.info("DASHBOARD AGI SHUTTING DOWN")
        self.logger.info("=" * 80)
        
        self.running = False
        
        # Cleanup resources
        if self.vision:
            self.vision.cleanup()
        
        # Wait for threads
        if self.obd_thread:
            self.obd_thread.join(timeout=2)
        if self.vision_thread:
            self.vision_thread.join(timeout=2)
        
        self.logger.info("Shutdown complete")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Main entry point."""
    # Load configuration
    config = DashboardConfig()
    
    # Create and run AGI system
    agi = DashboardAGI(config)
    agi.run()


if __name__ == "__main__":
    main()