"""
STDF Parser Module
Parses Standard Test Data Format (STDF) files to extract wafer test data.
"""

import struct
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WaferData:
    """Container for wafer test data."""
    wafer_id: str
    lot_id: str
    wafer_num: int
    die_count: int
    pass_count: int
    fail_count: int
    coordinates: np.ndarray  # (N, 2) array of (x, y) coordinates
    bins: np.ndarray  # (N,) array of bin assignments
    test_results: Optional[Dict[str, np.ndarray]] = None  # Test name -> values
    metadata: Optional[Dict] = None


class STDFParser:
    """Parser for STDF (Standard Test Data Format) files."""
    
    # STDF Record Types
    FAR = (0, 10)  # File Attributes Record
    MIR = (1, 10)  # Master Information Record
    MRR = (1, 20)  # Master Results Record
    WIR = (2, 10)  # Wafer Information Record
    WRR = (2, 20)  # Wafer Results Record
    PIR = (5, 10)  # Part Information Record
    PRR = (5, 20)  # Part Results Record
    PTR = (15, 10)  # Parametric Test Record
    FTR = (15, 20)  # Functional Test Record
    
    def __init__(self, stdf_path: str):
        """
        Initialize STDF parser.
        
        Args:
            stdf_path: Path to STDF file
        """
        self.stdf_path = Path(stdf_path)
        if not self.stdf_path.exists():
            raise FileNotFoundError(f"STDF file not found: {stdf_path}")
        
        self.lot_id: Optional[str] = None
        self.wafer_num: Optional[int] = None
        self.wafer_id: Optional[str] = None
        self.parts: List[Dict] = []
        
    def parse(self) -> WaferData:
        """
        Parse STDF file and extract wafer data.
        
        Returns:
            WaferData object containing parsed data
        """
        logger.info(f"Parsing STDF file: {self.stdf_path}")
        
        with open(self.stdf_path, 'rb') as f:
            while True:
                # Read record header (4 bytes: length, type, subtype)
                header = f.read(4)
                if not header or len(header) < 4:
                    break
                
                rec_len = struct.unpack('<H', header[:2])[0]
                rec_type = header[2]
                rec_subtype = header[3]
                
                # Read record data
                data = f.read(rec_len) if rec_len > 0 else b''
                
                # Process specific record types
                if (rec_type, rec_subtype) == self.MIR:
                    self._parse_mir(data)
                elif (rec_type, rec_subtype) == self.WIR:
                    self._parse_wir(data)
                elif (rec_type, rec_subtype) == self.PRR:
                    self._parse_prr(data)
                elif (rec_type, rec_subtype) == self.PTR:
                    self._parse_ptr(data)
        
        # Convert parts list to structured data
        wafer_data = self._build_wafer_data()
        logger.info(f"Parsed {len(self.parts)} dies from wafer {self.wafer_id}")
        
        return wafer_data
    
    def _parse_mir(self, data: bytes):
        """Parse Master Information Record."""
        try:
            # Extract lot ID (simplified - actual STDF format is more complex)
            # This is a placeholder - use pystdf library for production
            self.lot_id = f"LOT_{data[10:20].decode('ascii', errors='ignore').strip()}"
        except Exception as e:
            logger.warning(f"Error parsing MIR: {e}")
            self.lot_id = "UNKNOWN_LOT"
    
    def _parse_wir(self, data: bytes):
        """Parse Wafer Information Record."""
        try:
            # Extract wafer number
            self.wafer_num = struct.unpack('<B', data[0:1])[0] if len(data) >= 1 else 0
            self.wafer_id = f"{self.lot_id}-W{self.wafer_num:03d}"
        except Exception as e:
            logger.warning(f"Error parsing WIR: {e}")
            self.wafer_num = 0
            self.wafer_id = f"{self.lot_id}-W000"
    
    def _parse_prr(self, data: bytes):
        """Parse Part Result Record (die-level results)."""
        try:
            if len(data) < 10:
                return
            
            # Extract die coordinates and bin
            # Simplified parsing - actual STDF format requires proper field extraction
            x_coord = struct.unpack('<h', data[0:2])[0]
            y_coord = struct.unpack('<h', data[2:4])[0]
            hard_bin = struct.unpack('<H', data[4:6])[0]
            soft_bin = struct.unpack('<H', data[6:8])[0]
            
            # Part flag: bit 3 indicates pass/fail
            part_flag = data[8] if len(data) > 8 else 0
            is_pass = not (part_flag & 0x08)  # Bit 3: 0=pass, 1=fail
            
            self.parts.append({
                'x': x_coord,
                'y': y_coord,
                'bin': soft_bin if soft_bin > 0 else hard_bin,
                'pass_fail': 'PASS' if is_pass else 'FAIL'
            })
            
        except Exception as e:
            logger.warning(f"Error parsing PRR: {e}")
    
    def _parse_ptr(self, data: bytes):
        """Parse Parametric Test Record (test measurements)."""
        # Placeholder for parametric data extraction
        # In production, extract test names and values
        pass
    
    def _build_wafer_data(self) -> WaferData:
        """Build WaferData object from parsed parts."""
        if not self.parts:
            raise ValueError("No die data found in STDF file")
        
        # Convert to numpy arrays
        coordinates = np.array([[p['x'], p['y']] for p in self.parts])
        bins = np.array([p['bin'] for p in self.parts])
        
        # Count pass/fail
        pass_count = sum(1 for p in self.parts if p['pass_fail'] == 'PASS')
        fail_count = len(self.parts) - pass_count
        
        return WaferData(
            wafer_id=self.wafer_id or "UNKNOWN",
            lot_id=self.lot_id or "UNKNOWN",
            wafer_num=self.wafer_num or 0,
            die_count=len(self.parts),
            pass_count=pass_count,
            fail_count=fail_count,
            coordinates=coordinates,
            bins=bins,
            metadata={
                'yield': (pass_count / len(self.parts)) * 100 if self.parts else 0,
                'stdf_path': str(self.stdf_path)
            }
        )


def parse_stdf_file(stdf_path: str) -> WaferData:
    """
    Convenience function to parse STDF file.
    
    Args:
        stdf_path: Path to STDF file
        
    Returns:
        WaferData object
    """
    parser = STDFParser(stdf_path)
    return parser.parse()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python stdf_parser.py <stdf_file>")
        sys.exit(1)
    
    wafer_data = parse_stdf_file(sys.argv[1])
    print(f"Wafer: {wafer_data.wafer_id}")
    print(f"Lot: {wafer_data.lot_id}")
    print(f"Dies: {wafer_data.die_count}")
    print(f"Yield: {wafer_data.metadata['yield']:.2f}%")
