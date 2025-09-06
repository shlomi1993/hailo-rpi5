"""
HAILO Device Manager

Manages HAILO AI HAT device initialization, configuration, and lifecycle.
Provides high-level interface for device management using PyHailoRT.
"""
import logging

from typing import List, Optional, Dict, Any
from pathlib import Path

try:
    import hailo_platform.pyhailort.pyhailort as hailort
    from hailo_platform.pyhailort import HEF, Device, VDevice, ConfigureParams
    HAILO_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PyHailoRT not available: {e}")
    HAILO_AVAILABLE = False


class HailoDeviceManager:
    """
    High-level manager for HAILO AI HAT devices.

    Handles device discovery, initialization, and configuration using PyHailoRT API.
    """
    def __init__(self):
        self.device: Optional[Device] = None
        self.vdevice: Optional[VDevice] = None
        self.hef: Optional[HEF] = None
        self.network_groups = {}
        self.logger = logging.getLogger(__name__)

        if not HAILO_AVAILABLE:
            raise RuntimeError("PyHailoRT is not available. Please install HAILO platform.")

    def discover_devices(self) -> List[str]:
        """
        Discover available HAILO devices.

        Returns:
            List of device identifiers
        """
        try:
            devices = Device.scan()  # Scan for PCIe devices (typical for AI HAT)
            device_ids = [str(device.device_id) for device in devices]
            self.logger.info(f"Found {len(device_ids)} HAILO devices: {device_ids}")
            return device_ids
        except Exception as e:
            self.logger.error(f"Error discovering devices: {e}")
            return []

    def initialize_device(self, device_id: Optional[str] = None) -> bool:
        """
        Initialize HAILO device.

        Args:
            device_id: Specific device ID to use, or None for first available

        Returns:
            True if successful, False otherwise
        """
        try:
            if device_id:
                self.device = Device(device_id)
            else:
                # Use first available device
                devices = Device.scan()
                if not devices:
                    self.logger.error("No HAILO devices found")
                    return False
                self.device = devices[0]

            self.logger.info(f"Initialized device: {self.device.device_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing device: {e}")
            return False

    def create_vdevice(self, device_ids: Optional[List[str]] = None) -> bool:
        """
        Create virtual device for multi-device scenarios.

        Args:
            device_ids: List of device IDs to include in vdevice

        Returns:
            True if successful, False otherwise
        """
        try:
            self.vdevice = VDevice(device_ids)
            self.logger.info("Created virtual device")
            return True
        except Exception as e:
            self.logger.error(f"Error creating vdevice: {e}")
            return False

    def load_hef(self, hef_path: str) -> bool:
        """
        Load HEF (HAILO Executable Format) model file.

        Args:
            hef_path: Path to the .hef model file

        Returns:
            True if successful, False otherwise
        """
        try:
            hef_file = Path(hef_path)
            if not hef_file.exists():
                self.logger.error(f"HEF file not found: {hef_path}")
                return False

            self.hef = HEF(str(hef_file))
            self.logger.info(f"Loaded HEF: {hef_path}")

            network_groups = self.hef.get_network_groups()
            for ng in network_groups:
                self.logger.info(f"Available network group: {ng.name}")

            return True

        except Exception as e:
            self.logger.error(f"Error loading HEF: {e}")
            return False

    def configure_network_group(self, network_group_name: Optional[str] = None) -> bool:
        """
        Configure a network group for inference.

        Args:
            network_group_name: Name of network group to configure, or None for first

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.hef:
                self.logger.error("No HEF loaded")
                return False

            target_device = self.vdevice if self.vdevice else self.device
            if not target_device:
                self.logger.error("No device initialized")
                return False

            # Get network groups
            network_groups = self.hef.get_network_groups()
            if not network_groups:
                self.logger.error("No network groups found in HEF")
                return False

            # Select network group
            if network_group_name:
                selected_ng = next((ng for ng in network_groups if ng.name == network_group_name), None)
                if not selected_ng:
                    self.logger.error(f"Network group '{network_group_name}' not found")
                    return False
            else:
                selected_ng = network_groups[0]

            # Configure the network group
            configure_params = ConfigureParams.create_from_hef(self.hef, interface=hailort.HailoStreamInterface.PCIe)
            configured_ng = target_device.configure(self.hef, configure_params)[selected_ng.name]

            self.network_groups[selected_ng.name] = configured_ng
            self.logger.info(f"Configured network group: {selected_ng.name}")

            return True

        except Exception as e:
            self.logger.error(f"Error configuring network group: {e}")
            return False

    def get_device_info(self) -> Dict[str, Any]:
        """
        Get information about the initialized device.

        Returns:
            Dictionary containing device information
        """
        info = dict()

        if self.device:
            try:
                device_info = self.device.get_device_info()
                info.update({
                    'device_id': str(self.device.device_id),
                    'device_architecture': str(device_info.device_architecture),
                    'firmware_version': device_info.firmware_version,
                    'logger_version': device_info.logger_version,
                    'device_type': 'Physical Device'
                })
            except Exception as e:
                self.logger.error(f"Error getting device info: {e}")

        if self.vdevice:
            info['device_type'] = 'Virtual Device'

        if self.hef:
            try:
                network_groups = self.hef.get_network_groups()
                info['network_groups'] = [ng.name for ng in network_groups]
            except Exception as e:
                self.logger.error(f"Error getting HEF info: {e}")

        return info

    def cleanup(self):
        try:
            if self.network_groups:
                for ng_name, ng in self.network_groups.items():
                    try:
                        ng.shutdown()
                        self.logger.info(f"Shutdown network group: {ng_name}")
                    except Exception as e:
                        self.logger.error(f"Error shutting down network group {ng_name}: {e}")

                self.network_groups.clear()

            if self.device:
                self.device = None
                self.logger.info("Released device")

            if self.vdevice:
                self.vdevice = None
                self.logger.info("Released vdevice")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
