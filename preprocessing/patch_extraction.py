"""
Patch Extraction Utilities for Whole Slide Images

This module provides utilities for extracting patches from WSI files
for downstream analysis with CPath-Omni.
"""

import numpy as np
from typing import List, Tuple, Optional, Generator, Union
from pathlib import Path
from PIL import Image
from dataclasses import dataclass

try:
    import openslide
except ImportError:
    openslide = None

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class Patch:
    """Container for an extracted patch with metadata."""
    image: np.ndarray
    x: int
    y: int
    level: int
    patch_size: int
    tissue_fraction: float
    slide_name: str


class PatchExtractor:
    """
    Extract patches from Whole Slide Images (WSI).
    
    Supports various WSI formats via OpenSlide (.svs, .tiff, .ndpi, etc.).
    
    Args:
        patch_size: Size of patches to extract (default: 1024)
        stride: Stride between patches (default: 512)
        level: Pyramid level to extract from (default: 0 = highest resolution)
        tissue_threshold: Minimum tissue fraction to include patch (default: 0.1)
    
    Example:
        >>> extractor = PatchExtractor(patch_size=1024, stride=512)
        >>> patches = extractor.extract("slide.svs")
        >>> for patch in patches:
        ...     print(f"Patch at ({patch.x}, {patch.y}): {patch.tissue_fraction:.2%} tissue")
    """
    
    def __init__(
        self,
        patch_size: int = 1024,
        stride: int = 512,
        level: int = 0,
        tissue_threshold: float = 0.1
    ):
        if openslide is None:
            raise ImportError("openslide-python is required. Install with: pip install openslide-python")
        if cv2 is None:
            raise ImportError("opencv-python is required. Install with: pip install opencv-python")
        
        self.patch_size = patch_size
        self.stride = stride
        self.level = level
        self.tissue_threshold = tissue_threshold
    
    def _detect_tissue(self, patch: np.ndarray) -> float:
        """
        Detect tissue fraction in a patch using simple thresholding.
        
        Args:
            patch: RGB patch as numpy array
        
        Returns:
            Fraction of pixels that are tissue (0-1)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        
        # Threshold: tissue is darker than background
        _, tissue_mask = cv2.threshold(gray, 200, 1, cv2.THRESH_BINARY_INV)
        
        tissue_fraction = tissue_mask.sum() / tissue_mask.size
        return tissue_fraction
    
    def extract(
        self,
        wsi_path: Union[str, Path],
        max_patches: Optional[int] = None
    ) -> List[Patch]:
        """
        Extract all patches from a WSI.
        
        Args:
            wsi_path: Path to WSI file
            max_patches: Maximum number of patches to extract (None = all)
        
        Returns:
            List of Patch objects
        """
        return list(self.extract_generator(wsi_path, max_patches))
    
    def extract_generator(
        self,
        wsi_path: Union[str, Path],
        max_patches: Optional[int] = None
    ) -> Generator[Patch, None, None]:
        """
        Extract patches as a generator (memory efficient).
        
        Args:
            wsi_path: Path to WSI file
            max_patches: Maximum number of patches to extract
        
        Yields:
            Patch objects
        """
        wsi_path = Path(wsi_path)
        slide = openslide.OpenSlide(str(wsi_path))
        
        width, height = slide.level_dimensions[self.level]
        downsample = slide.level_downsamples[self.level]
        
        patch_count = 0
        
        for x in range(0, width - self.patch_size, self.stride):
            for y in range(0, height - self.patch_size, self.stride):
                # Read patch
                # Note: read_region uses level 0 coordinates
                x0 = int(x * downsample)
                y0 = int(y * downsample)
                
                patch_img = slide.read_region(
                    (x0, y0), 
                    self.level, 
                    (self.patch_size, self.patch_size)
                ).convert('RGB')
                patch_np = np.array(patch_img)
                
                # Check tissue content
                tissue_fraction = self._detect_tissue(patch_np)
                
                if tissue_fraction >= self.tissue_threshold:
                    yield Patch(
                        image=patch_np,
                        x=x0,
                        y=y0,
                        level=self.level,
                        patch_size=self.patch_size,
                        tissue_fraction=tissue_fraction,
                        slide_name=wsi_path.name
                    )
                    
                    patch_count += 1
                    if max_patches is not None and patch_count >= max_patches:
                        slide.close()
                        return
        
        slide.close()
    
    def get_slide_info(self, wsi_path: Union[str, Path]) -> dict:
        """
        Get information about a WSI.
        
        Args:
            wsi_path: Path to WSI file
        
        Returns:
            Dictionary with slide information
        """
        wsi_path = Path(wsi_path)
        slide = openslide.OpenSlide(str(wsi_path))
        
        info = {
            "path": str(wsi_path),
            "name": wsi_path.name,
            "dimensions": slide.dimensions,
            "level_count": slide.level_count,
            "level_dimensions": slide.level_dimensions,
            "level_downsamples": slide.level_downsamples,
            "properties": dict(slide.properties),
        }
        
        # Estimate patch count
        width, height = slide.level_dimensions[self.level]
        n_x = (width - self.patch_size) // self.stride + 1
        n_y = (height - self.patch_size) // self.stride + 1
        info["estimated_patches"] = n_x * n_y
        
        slide.close()
        return info


def extract_patches_from_wsi(
    wsi_path: Union[str, Path],
    patch_size: int = 1024,
    stride: int = 512,
    tissue_threshold: float = 0.1,
    max_patches: Optional[int] = None
) -> List[Patch]:
    """
    Convenience function to extract patches from a WSI.
    
    Args:
        wsi_path: Path to WSI file
        patch_size: Size of patches
        stride: Stride between patches
        tissue_threshold: Minimum tissue fraction
        max_patches: Maximum patches to extract
    
    Returns:
        List of Patch objects
    """
    extractor = PatchExtractor(
        patch_size=patch_size,
        stride=stride,
        tissue_threshold=tissue_threshold
    )
    return extractor.extract(wsi_path, max_patches)


def create_tissue_mask(
    wsi_path: Union[str, Path],
    level: int = -1,
    threshold: int = 200
) -> Tuple[np.ndarray, float]:
    """
    Create a tissue mask for a WSI at a given level.
    
    Args:
        wsi_path: Path to WSI file
        level: Pyramid level (-1 = lowest resolution)
        threshold: Grayscale threshold for tissue detection
    
    Returns:
        Tuple of (binary mask, downsample factor)
    """
    if openslide is None:
        raise ImportError("openslide-python is required")
    
    slide = openslide.OpenSlide(str(wsi_path))
    
    if level < 0:
        level = slide.level_count + level
    
    # Read thumbnail at specified level
    dims = slide.level_dimensions[level]
    thumbnail = slide.read_region((0, 0), level, dims).convert('RGB')
    thumbnail_np = np.array(thumbnail)
    
    # Create mask
    gray = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Clean up with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    downsample = slide.level_downsamples[level]
    slide.close()
    
    return mask, downsample
