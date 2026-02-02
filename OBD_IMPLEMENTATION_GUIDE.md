# OBD-II Implementation Guide

This document provides guidance for implementing actual OBD-II hardware integration with the Dashboard AGI system.

## Overview

The current `OBDInterface` class includes placeholder functions for OBD-II communication. This guide shows how to integrate real OBD-II adapters.

## Recommended Libraries

### python-OBD (Recommended)

```bash
pip install obd
```

**Pros:**
- Well-maintained
- Supports ELM327 adapters
- Automatic command detection
- Async support

**Example Implementation:**

```python
import obd

class OBDInterface:
    def _initialize_connection(self):
        """Initialize real OBD-II connection."""
        try:
            # Auto-detect port
            self.connection = obd.OBD()
            
            # Or specify port
            # self.connection = obd.OBD(portstr=self.config.OBD_PORT)
            
            self.is_connected = self.connection.is_connected()
            
            if self.is_connected:
                self.logger.info(f"OBD-II connected: {self.connection.port_name()}")
                self.logger.info(f"Protocol: {self.connection.protocol_name()}")
            else:
                self.logger.error("Failed to connect to OBD-II")
                
        except Exception as e:
            self.logger.error(f"OBD-II initialization failed: {e}")
            self.is_connected = False
    
    def read_data(self) -> OBDData:
        """Read real OBD-II data."""
        if not self.is_connected:
            return self._get_simulated_data()
        
        try:
            data = OBDData(timestamp=datetime.now())
            
            # Read PIDs (Parameter IDs)
            rpm = self.connection.query(obd.commands.RPM)
            if not rpm.is_null():
                data.engine_rpm = rpm.value.magnitude
            
            speed = self.connection.query(obd.commands.SPEED)
            if not speed.is_null():
                data.vehicle_speed = speed.value.magnitude
            
            temp = self.connection.query(obd.commands.COOLANT_TEMP)
            if not temp.is_null():
                data.engine_temp = temp.value.magnitude
            
            throttle = self.connection.query(obd.commands.THROTTLE_POS)
            if not throttle.is_null():
                data.throttle_position = throttle.value.magnitude
            
            fuel = self.connection.query(obd.commands.FUEL_LEVEL)
            if not fuel.is_null():
                data.fuel_level = fuel.value.magnitude
            
            voltage = self.connection.query(obd.commands.CONTROL_MODULE_VOLTAGE)
            if not voltage.is_null():
                data.battery_voltage = voltage.value.magnitude
            
            intake = self.connection.query(obd.commands.INTAKE_TEMP)
            if not intake.is_null():
                data.intake_air_temp = intake.value.magnitude
            
            load = self.connection.query(obd.commands.ENGINE_LOAD)
            if not load.is_null():
                data.engine_load = load.value.magnitude
            
            # Read DTCs
            dtc_response = self.connection.query(obd.commands.GET_DTC)
            if not dtc_response.is_null():
                data.dtc_codes = [dtc[0] for dtc in dtc_response.value]
            
            self.last_data = data
            self.data_history.append(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error reading OBD data: {e}")
            return OBDData(timestamp=datetime.now())
```

### PySerial (Low-level)

For custom implementations:

```bash
pip install pyserial
```

**Example:**

```python
import serial

class OBDInterface:
    def _initialize_connection(self):
        """Direct serial connection to ELM327."""
        try:
            self.serial = serial.Serial(
                port=self.config.OBD_PORT,
                baudrate=self.config.OBD_BAUDRATE,
                timeout=1
            )
            
            # Initialize ELM327
            self._send_command("ATZ")  # Reset
            self._send_command("ATE0")  # Echo off
            self._send_command("ATL0")  # Line feeds off
            self._send_command("ATSP0")  # Auto protocol
            
            self.is_connected = True
            self.logger.info("ELM327 initialized")
            
        except Exception as e:
            self.logger.error(f"Serial connection failed: {e}")
            self.is_connected = False
    
    def _send_command(self, cmd: str) -> str:
        """Send AT/OBD command and return response."""
        self.serial.write((cmd + '\r').encode())
        time.sleep(0.1)
        response = self.serial.read(self.serial.in_waiting).decode()
        return response.strip()
    
    def _query_pid(self, pid: str) -> Optional[str]:
        """Query specific PID."""
        response = self._send_command(pid)
        if "NO DATA" in response or "?" in response:
            return None
        return response
    
    def read_data(self) -> OBDData:
        """Read OBD data using PIDs."""
        data = OBDData(timestamp=datetime.now())
        
        try:
            # Mode 01 PIDs
            # 010C = Engine RPM
            rpm_raw = self._query_pid("010C")
            if rpm_raw:
                # Parse hex response: ((A*256)+B)/4
                bytes_data = bytes.fromhex(rpm_raw.replace(' ', ''))
                data.engine_rpm = ((bytes_data[2] * 256) + bytes_data[3]) / 4
            
            # 010D = Vehicle Speed
            speed_raw = self._query_pid("010D")
            if speed_raw:
                bytes_data = bytes.fromhex(speed_raw.replace(' ', ''))
                data.vehicle_speed = bytes_data[2]
            
            # 0105 = Engine Coolant Temperature
            temp_raw = self._query_pid("0105")
            if temp_raw:
                bytes_data = bytes.fromhex(temp_raw.replace(' ', ''))
                data.engine_temp = bytes_data[2] - 40
            
            # Add more PIDs as needed...
            
        except Exception as e:
            self.logger.error(f"PID query error: {e}")
        
        return data
```

## Common OBD-II PIDs

### Mode 01 - Current Data

| PID | Description | Formula | Unit |
|-----|-------------|---------|------|
| 010C | Engine RPM | ((A*256)+B)/4 | rpm |
| 010D | Vehicle Speed | A | km/h |
| 0105 | Coolant Temp | A-40 | °C |
| 0111 | Throttle Position | A*100/255 | % |
| 010F | Intake Air Temp | A-40 | °C |
| 0104 | Engine Load | A*100/255 | % |
| 012F | Fuel Level | A*100/255 | % |
| 0142 | Control Module Voltage | ((A*256)+B)/1000 | V |
| 0110 | MAF Air Flow | ((A*256)+B)/100 | g/s |

### Mode 03 - Diagnostic Trouble Codes

```python
# Request DTCs
dtc_response = self._send_command("03")

# Parse DTC codes
# Each DTC is 2 bytes in hex format
# Example: "430100" = P0100 (MAF Circuit Malfunction)
```

### DTC Format

- **Pxxxx**: Powertrain
- **Cxxxx**: Chassis
- **Bxxxx**: Body
- **Uxxxx**: Network

## Hardware Recommendations

### USB OBD-II Adapters

1. **OBDLink SX** (Recommended)
   - Professional grade
   - High compatibility
   - Fast response times
   - ~$50

2. **ELM327 USB**
   - Budget option
   - Variable quality
   - Check chipset (genuine vs. clone)
   - ~$10-30

3. **BAFX Products**
   - Good quality
   - Well-supported
   - ~$25

### Bluetooth OBD-II Adapters

1. **BAFX Products 34t5**
   - Wide compatibility
   - Android & Linux support
   - ~$25

2. **OBDLink MX+**
   - Professional features
   - iOS & Android
   - ~$100

**Note:** For Raspberry Pi, USB adapters are more reliable than Bluetooth.

## Connection Troubleshooting

### 1. Port Detection

```bash
# Linux - List serial devices
ls -l /dev/tty*

# Common OBD ports:
# /dev/ttyUSB0 (USB adapter)
# /dev/rfcomm0 (Bluetooth)

# Check permissions
sudo usermod -a -G dialout $USER
```

### 2. Baud Rate

Most ELM327 adapters use **38400** baud, but some use:
- 9600
- 115200

Try different rates if connection fails:

```python
for baudrate in [38400, 9600, 115200]:
    try:
        conn = serial.Serial(port, baudrate=baudrate, timeout=1)
        # Test connection
        conn.write(b'ATZ\r')
        response = conn.read(100)
        if b'ELM' in response:
            print(f"Connected at {baudrate} baud")
            break
    except:
        continue
```

### 3. Protocol Detection

```python
# Let ELM327 auto-detect protocol
self._send_command("ATSP0")

# Or manually set:
# 0 = Auto
# 1 = SAE J1850 PWM
# 2 = SAE J1850 VPW
# 3 = ISO 9141-2
# 4 = ISO 14230-4 (KWP2000 5-baud)
# 5 = ISO 14230-4 (KWP2000 fast)
# 6 = ISO 15765-4 (CAN 11/500)
# 7 = ISO 15765-4 (CAN 29/500)
# 8 = ISO 15765-4 (CAN 11/250)
# 9 = ISO 15765-4 (CAN 29/250)
```

## Error Handling

```python
def read_data_with_retry(self, max_retries=3) -> OBDData:
    """Read OBD data with automatic retry."""
    for attempt in range(max_retries):
        try:
            data = self.read_data()
            if data.engine_rpm is not None:  # Validation
                return data
        except Exception as e:
            self.logger.warning(f"Read attempt {attempt + 1} failed: {e}")
            time.sleep(0.5)
    
    self.logger.error("All read attempts failed")
    return self._get_simulated_data()
```

## Advanced Features

### Async OBD Reading

```python
import asyncio
import obd

async def read_obd_async(self):
    """Non-blocking OBD reads."""
    connection = obd.Async(self.config.OBD_PORT)
    
    # Register callbacks
    def rpm_callback(response):
        if not response.is_null():
            self.last_rpm = response.value.magnitude
    
    connection.watch(obd.commands.RPM, callback=rpm_callback)
    connection.start()
    
    # Run in background
    while self.running:
        await asyncio.sleep(self.config.OBD_POLL_INTERVAL)
```

### Custom PIDs

```python
# Define custom PID
custom_pid = obd.OBDCommand(
    "Custom_PID",
    "Description",
    "01AB",  # Mode and PID
    2,       # Number of return bytes
    lambda messages: parse_custom(messages)
)

# Use it
response = connection.query(custom_pid)
```

## Testing Without Vehicle

### Software Emulator

Use an OBD-II simulator for development:

1. **OBDSim** (Linux)
```bash
sudo apt install obdsim
obdsim -g gui
```

2. **Virtual Serial Ports**
```bash
# Linux
socat -d -d pty,raw,echo=0 pty,raw,echo=0
# Creates pair of virtual serial ports
```

## Production Checklist

- [ ] Test with actual vehicle
- [ ] Verify all PIDs supported by your vehicle
- [ ] Handle disconnections gracefully
- [ ] Implement reconnection logic
- [ ] Add data validation
- [ ] Log communication errors
- [ ] Test protocol auto-detection
- [ ] Verify DTC reading
- [ ] Test in various driving conditions
- [ ] Monitor CPU/memory usage

## Safety Considerations

⚠️ **Important:**

1. **Never write to OBD-II** unless you know exactly what you're doing
2. **Read-only operations** are safe
3. **Avoid excessive polling** (< 1 second intervals may cause issues)
4. **Don't interfere with critical systems**
5. **Test thoroughly before production use**

## References

- [python-OBD Documentation](https://python-obd.readthedocs.io/)
- [OBD-II PIDs Wikipedia](https://en.wikipedia.org/wiki/OBD-II_PIDs)
- [ELM327 Datasheet](https://www.elmelectronics.com/wp-content/uploads/2017/01/ELM327DS.pdf)
- [ISO 15765-4 (CAN) Standard](https://www.iso.org/standard/66369.html)

---

For questions or issues, refer to the main README or open an issue.
