"""
Wafer Map Generator
Converts STDF die-level data to 300x300 RGB wafer map images.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
from PIL import Image, ImageDraw
import cv2

from .stdf_parser import WaferData

logger = logging.getLogger(__name__)


class WaferMapGenerator:
    """Generates wafer map images from die-level test data."""
    
    # Color scheme for bins
    COLORS = {
        'PASS': (0, 255, 0),      # Green
        'FAIL_BIN2': (255, 0, 0),  # Red
        'FAIL_BIN3': (255, 140, 0),  # Orange
        'FAIL_BIN4': (255, 215, 0),  # Yellow
        'FAIL_BIN5': (255, 69, 0),   # Red-Orange
        'FAIL_BIN6': (220, 20, 60),  # Crimson
        'FAIL_BIN7': (178, 34, 34),  # Fire Brick
        'FAIL_BIN8': (139, 0, 0),    # Dark Red
        'FAIL_BIN9': (128, 0, 0),    # Maroon
        'NOTEST': (128, 128, 128)   # Gray
    }
    
    def __init__(
        self,
        image_size: int = 300,
        die_size: int = 3,
        background_color: Tuple[int, int, int] = (0, 0, 0)
    ):
        """
        Initialize wafer map generator.
        
        Args:
            image_size: Output image size (width and height)
            die_size: Size of each die in pixels
            background_color: Background color (R, G, B)
        """
        self.image_size = image_size
        self.die_size = die_size
        self.background_color = background_color
        
    def generate(
        self,
        wafer_data: WaferData,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate wafer map image from wafer data.
        
        Args:
            wafer_data: WaferData object from STDF parser
            output_path: Optional path to save image
            
        Returns:
            RGB image as numpy array (H, W, 3)
        """
        logger.info(f"Generating wafer map for {wafer_data.wafer_id}")
        
        # Normalize coordinates to image space
        coords_norm = self._normalize_coordinates(
            wafer_data.coordinates,
            self.image_size,
            self.die_size
        )
        
        # Create blank image
        image = np.full(
            (self.image_size, self.image_size, 3),
            self.background_color,
            dtype=np.uint8
        )
        
        # Draw each die
        for (x, y), bin_num in zip(coords_norm, wafer_data.bins):
            color = self._get_bin_color(bin_num)
            self._draw_die(image, x, y, color)
        
        # Optional: Add circular wafer boundary
        image = self._add_wafer_boundary(image)
        
        # Save if output path provided
        if output_path:
            self._save_image(image, output_path)
            logger.info(f"Saved wafer map to {output_path}")
        
        return image
    
    def _normalize_coordinates(
        self,
        coords: np.ndarray,
        image_size: int,
        die_size: int
    ) -> np.ndarray:
        """
        Normalize die coordinates to image pixel space.
        
        Args:
            coords: (N, 2) array of (x, y) coordinates
            image_size: Target image size
            die_size: Die size in pixels
            
        Returns:
            Normalized coordinates (N, 2)
        """
        if len(coords) == 0:
            return np.array([])
        
        # Find bounding box
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        
        # Calculate scaling factor
        x_range = x_max - x_min
        y_range = y_max - y_min
        max_range = max(x_range, y_range)
        
        if max_range == 0:
            scale = 1
        else:
            # Leave margin for dies at edges
            scale = (image_size - 2 * die_size) / max_range
        
        # Normalize and center
        coords_norm = (coords - [x_min, y_min]) * scale
        
        # Center in image
        offset_x = (image_size - (x_range * scale)) / 2
        offset_y = (image_size - (y_range * scale)) / 2
        coords_norm += [offset_x, offset_y]
        
        return coords_norm.astype(int)
    
    def _get_bin_color(self, bin_num: int) -> Tuple[int, int, int]:
        """
        Get color for bin number.
        
        Args:
            bin_num: Bin number (1=PASS, 2-9=FAIL)
            
        Returns:
            RGB color tuple
        """
        if bin_num == 1:
            return self.COLORS['PASS']
        elif bin_num == 0:
            return self.COLORS['NOTEST']
        elif bin_num <= 9:
            key = f'FAIL_BIN{bin_num}'
            return self.COLORS.get(key, self.COLORS['FAIL_BIN2'])
        else:
            # For bins > 9, use red
            return self.COLORS['FAIL_BIN2']
    
    def _draw_die(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        color: Tuple[int, int, int]
    ):
        """
        Draw a single die on the image.
        
        Args:
            image: Image array to draw on
            x, y: Die center coordinates
            color: RGB color
        """
        half_size = self.die_size // 2
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(self.image_size - 1, x + half_size)
        y2 = min(self.image_size - 1, y + half_size)
        
        image[y1:y2, x1:x2] = color
    
    def _add_wafer_boundary(self, image: np.ndarray) -> np.ndarray:
        """
        Add circular wafer boundary to image.
        
        Args:
            image: Input image
            
        Returns:
            Image with boundary added
        """
        # Create circular mask
        center = (self.image_size // 2, self.image_size // 2)
        radius = self.image_size // 2 - 5
        
        # Draw circle outline
        cv2.circle(
            image,
            center,
            radius,
            (255, 255, 255),  # White
            thickness=2
        )
        
        return image
    
    def _save_image(self, image: np.ndarray, output_path: str):
        """Save image to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert BGR to RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        pil_image.save(str(output_path), format='PNG', optimize=True)


def generate_wafer_map(
    wafer_data: WaferData,
    output_path: Optional[str] = None,
    image_size: int = 300
) -> np.ndarray:
    """
    Convenience function to generate wafer map.
    
    Args:
        wafer_data: WaferData object
        output_path: Optional save path
        image_size: Output image size
        
    Returns:
        RGB image array
    """
    generator = WaferMapGenerator(image_size=image_size)
    return generator.generate(wafer_data, output_path)


if __name__ == "__main__":
    # Example usage
    import sys
    from .stdf_parser import parse_stdf_file
    
    if len(sys.argv) < 3:
        print("Usage: python wafer_map_generator.py <stdf_file> <output_png>")
        sys.exit(1)
    
    wafer_data = parse_stdf_file(sys.argv[1])
    image = generate_wafer_map(wafer_data, sys.argv[2])
    print(f"Generated wafer map: {sys.argv[2]} ({image.shape})")
