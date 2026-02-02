#!/bin/bash
# Dashboard AGI System - Quick Start Setup Script
# This script automates the installation and configuration process

set -e  # Exit on error

echo "=================================="
echo "Dashboard AGI System Setup"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}Error: Python 3.8+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK: $PYTHON_VERSION${NC}"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install dependencies
echo ""
echo "Installing Python dependencies..."
echo "This may take a few minutes..."
pip install -r requirements_agi.txt
echo -e "${GREEN}✓ Python dependencies installed${NC}"

# Check for mpg123 (audio player)
echo ""
echo "Checking for audio player (mpg123)..."
if command -v mpg123 &> /dev/null; then
    echo -e "${GREEN}✓ mpg123 found${NC}"
else
    echo -e "${YELLOW}⚠ mpg123 not found${NC}"
    echo "Please install mpg123 for audio playback:"
    echo "  Ubuntu/Debian: sudo apt install mpg123"
    echo "  macOS: brew install mpg123"
fi

# Create .env file if it doesn't exist
echo ""
if [ -f ".env" ]; then
    echo -e "${YELLOW}.env file already exists${NC}"
else
    echo "Creating .env file..."
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created${NC}"
    echo -e "${YELLOW}⚠ IMPORTANT: Edit .env and add your OpenAI API key!${NC}"
fi

# Create output directory
echo ""
echo "Creating output directory..."
mkdir -p dashboard_data/camera_frames
echo -e "${GREEN}✓ Output directory created${NC}"

# Download YOLO model
echo ""
echo "Checking for YOLO model..."
if [ -f "yolov8n.pt" ]; then
    echo -e "${GREEN}✓ YOLO model already exists${NC}"
else
    echo "Downloading YOLOv8 nano model..."
    echo "(This will happen automatically on first run if skipped)"
    read -p "Download now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
        echo -e "${GREEN}✓ YOLO model downloaded${NC}"
    else
        echo "Skipping YOLO download (will auto-download on first run)"
    fi
fi

# Test camera
echo ""
echo "Testing camera..."
python3 << EOF
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("\033[0;32m✓ Camera detected and working\033[0m")
    cap.release()
else:
    print("\033[1;33m⚠ No camera detected (will use simulated data)\033[0m")
EOF

# Check for OBD-II device
echo ""
echo "Checking for OBD-II adapter..."
if [ -e "/dev/ttyUSB0" ]; then
    echo -e "${GREEN}✓ USB device detected at /dev/ttyUSB0${NC}"
    echo "This might be your OBD-II adapter"
    echo "Check permissions: ls -l /dev/ttyUSB0"
    echo "Add user to dialout group: sudo usermod -a -G dialout \$USER"
else
    echo -e "${YELLOW}⚠ No USB device at /dev/ttyUSB0${NC}"
    echo "OBD-II adapter not detected (will use simulated data)"
fi

# Final instructions
echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add your OpenAI API key:"
echo "   nano .env"
echo ""
echo "2. (Optional) Customize configuration:"
echo "   nano config.json"
echo ""
echo "3. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "4. Run the system:"
echo "   python dashboard_agi.py"
echo ""
echo "For OBD-II integration, see: OBD_IMPLEMENTATION_GUIDE.md"
echo "For full documentation, see: README_AGI.md"
echo ""
echo -e "${YELLOW}⚠ IMPORTANT:${NC}"
echo "- Ensure OPENAI_API_KEY is set in .env"
echo "- Test in safe environment before using in vehicle"
echo "- Camera and microphone access required"
echo ""
echo "Happy driving! 🚗"
