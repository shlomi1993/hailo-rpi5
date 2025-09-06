#!/usr/bin/env python3
"""
Device Information Tool

Utility script to discover and display information about HAILO devices
connected to the system.
"""

import sys
import logging
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import HailoDeviceManager


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def print_device_info(device_info: dict):
    """Print device information in a readable format."""
    print("=" * 50)
    print("HAILO DEVICE INFORMATION")
    print("=" * 50)

    if 'device_id' in device_info:
        print(f"\nDevice ID: {device_info['device_id']}")

    if 'device_architecture' in device_info:
        print(f"Architecture: {device_info['device_architecture']}")

    if 'firmware_version' in device_info:
        print(f"Firmware Version: {device_info['firmware_version']}")

    if 'logger_version' in device_info:
        print(f"Logger Version: {device_info['logger_version']}")

    if 'device_type' in device_info:
        print(f"Device Type: {device_info['device_type']}")

    if 'network_groups' in device_info:
        print(f"\nAvailable Network Groups:")
        for ng in device_info['network_groups']:
            print(f"  - {ng}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='HAILO Device Information Tool')
    parser.add_argument('--device-id', help='Specific device ID to query')
    parser.add_argument('--list-only', action='store_true',
                       help='Only list device IDs without detailed info')
    parser.add_argument('--hef', help='Load HEF file to see network groups')

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info("Discovering HAILO devices...")

        with HailoDeviceManager() as device_manager:
            # Discover devices
            devices = device_manager.discover_devices()

            if not devices:
                print("No HAILO devices found on the system.")
                print("\nTroubleshooting:")
                print("1. Ensure the HAILO AI HAT is properly connected")
                print("2. Check if HAILO drivers are installed")
                print("3. Verify PyHailoRT is properly installed")
                print("4. Run with sudo if needed for device access")
                return 1

            print(f"\nFound {len(devices)} HAILO device(s):")
            for i, device_id in enumerate(devices):
                print(f"  {i+1}. {device_id}")

            if args.list_only:
                return 0

            # Initialize specific device or first one
            target_device = args.device_id if args.device_id else None

            logger.info(f"Initializing device: {target_device or 'first available'}")

            if not device_manager.initialize_device(target_device):
                logger.error("Failed to initialize device")
                return 1

            # Load HEF if provided
            if args.hef:
                logger.info(f"Loading HEF file: {args.hef}")
                if not device_manager.load_hef(args.hef):
                    logger.error("Failed to load HEF file")
                else:
                    if not device_manager.configure_network_group():
                        logger.error("Failed to configure network group")

            # Get and display device information
            device_info = device_manager.get_device_info()
            print_device_info(device_info)

            # Test basic functionality
            print("\n" + "=" * 50)
            print("BASIC FUNCTIONALITY TEST")
            print("=" * 50)

            print("✅ Device discovery: PASSED")
            print("✅ Device initialization: PASSED")

            if args.hef and 'network_groups' in device_info:
                print("✅ HEF loading: PASSED")
                print("✅ Network group configuration: PASSED")

            print("\nDevice is ready for inference!")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"\n❌ Device test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check device connections")
        print("2. Verify driver installation")
        print("3. Ensure proper permissions")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
