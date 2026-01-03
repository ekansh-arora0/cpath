"""
Macenko Stain Normalization for H&E Histopathology Images

This module implements the Macenko method for stain normalization,
which normalizes the color appearance of H&E stained tissue sections
to a reference image.

Reference:
    Macenko, M., et al. "A method for normalizing histology slides for 
    quantitative analysis." 2009 IEEE International Symposium on 
    Biomedical Imaging.
"""

import numpy as np
from typing import Optional


class MacenkoNormalizer:
    """
    Macenko Stain Normalizer for H&E Images
    
    Normalizes histopathology images to match a reference staining appearance
    using the Macenko method based on singular value decomposition.
    
    Usage:
        >>> normalizer = MacenkoNormalizer()
        >>> normalizer.fit(reference_image)  # Set reference from target slide
        >>> normalized = normalizer.transform(source_image)
    
    Attributes:
        target_stain_matrix: Learned stain matrix from reference image
        target_concentrations: Max concentrations from reference image
        target_means: Mean OD values (for checking if fitted)
    """
    
    def __init__(self):
        self.target_stain_matrix: Optional[np.ndarray] = None
        self.target_concentrations: Optional[np.ndarray] = None
        self.target_means: Optional[np.ndarray] = None
    
    def _rgb2od(self, img: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to optical density (OD) space.
        
        Args:
            img: RGB image as numpy array (H, W, 3), uint8
        
        Returns:
            Optical density image (H, W, 3), float32
        """
        img = img.astype(np.float32)
        # Add 1 to avoid log(0), divide by 255 to normalize
        od = -np.log((img + 1) / 255)
        return od
    
    def _od2rgb(self, od: np.ndarray) -> np.ndarray:
        """
        Convert optical density back to RGB.
        
        Args:
            od: Optical density image (H, W, 3), float32
        
        Returns:
            RGB image (H, W, 3), uint8
        """
        rgb = (255 * np.exp(-od)).clip(0, 255).astype(np.uint8)
        return rgb
    
    def fit(self, target_img: np.ndarray) -> 'MacenkoNormalizer':
        """
        Learn stain matrix from reference/target image.
        
        This extracts the principal stain vectors (H and E) from the
        target image using SVD, which will be used to normalize other images.
        
        Args:
            target_img: Reference RGB image (H, W, 3), uint8
        
        Returns:
            self (for method chaining)
        """
        # Convert to OD and flatten
        od = self._rgb2od(target_img).reshape((-1, 3))
        
        # Remove background pixels (low OD values)
        od_valid = od[~np.any(od < 0.15, axis=1)]
        
        if od_valid.shape[0] < 100:
            raise ValueError("Not enough foreground pixels in reference image")
        
        # SVD to find principal stain vectors
        U, S, Vt = np.linalg.svd(od_valid, full_matrices=False)
        
        # First 2 principal directions are H and E stains
        stain_matrix = Vt[:2, :].T
        
        # Normalize columns to unit length
        stain_matrix /= np.linalg.norm(stain_matrix, axis=0)
        
        self.target_stain_matrix = stain_matrix
        
        # Get concentrations for scaling
        C = np.dot(od_valid, stain_matrix)
        maxC = np.percentile(C, 99, axis=0)
        self.target_concentrations = maxC
        
        # Store mean for checking if fitted
        self.target_means = od_valid.mean(axis=0)
        
        return self
    
    def transform(self, img: np.ndarray) -> np.ndarray:
        """
        Apply stain normalization to an image.
        
        Args:
            img: Source RGB image (H, W, 3), uint8
        
        Returns:
            Normalized RGB image (H, W, 3), uint8
        """
        if self.target_stain_matrix is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        original_shape = img.shape
        
        # Convert to OD
        od = self._rgb2od(img).reshape((-1, 3))
        
        # Get valid (non-background) pixels
        od_valid = od[~np.any(od < 0.15, axis=1)]
        
        if od_valid.shape[0] < 10:
            # Mostly background, return original
            return img
        
        # SVD on source image
        U, S, Vt = np.linalg.svd(od_valid, full_matrices=False)
        stain_matrix_source = Vt[:2, :].T
        stain_matrix_source /= np.linalg.norm(stain_matrix_source, axis=0)
        
        # Project all pixels onto source stain vectors
        C = np.dot(od, stain_matrix_source)
        
        # Rescale to match target concentrations
        source_maxC = np.percentile(C[~np.any(od < 0.15, axis=1)], 99, axis=0)
        
        # Avoid division by zero
        source_maxC = np.maximum(source_maxC, 1e-6)
        
        C *= (self.target_concentrations / source_maxC)
        
        # Reconstruct OD using target stain matrix
        od_normalized = np.dot(C, self.target_stain_matrix.T)
        od_normalized = od_normalized.reshape(original_shape)
        
        return self._od2rgb(od_normalized)
    
    def fit_transform(self, img: np.ndarray) -> np.ndarray:
        """
        Fit to image and transform it (useful for setting reference).
        
        Args:
            img: RGB image (H, W, 3), uint8
        
        Returns:
            Normalized image (same as input since it's the reference)
        """
        self.fit(img)
        return img
    
    @property
    def is_fitted(self) -> bool:
        """Check if normalizer has been fitted."""
        return self.target_stain_matrix is not None


class ReinhardNormalizer:
    """
    Reinhard Color Normalization in LAB Space
    
    An alternative to Macenko that normalizes in LAB color space
    by matching mean and standard deviation of each channel.
    
    Reference:
        Reinhard, E., et al. "Color transfer between images."
        IEEE Computer Graphics and Applications, 2001.
    """
    
    def __init__(self):
        self.target_means: Optional[np.ndarray] = None
        self.target_stds: Optional[np.ndarray] = None
    
    def fit(self, target_img: np.ndarray) -> 'ReinhardNormalizer':
        """
        Learn color statistics from reference image.
        
        Args:
            target_img: Reference RGB image (H, W, 3), uint8
        
        Returns:
            self
        """
        import cv2
        
        # Convert to LAB
        lab = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Compute statistics
        self.target_means = lab.mean(axis=(0, 1))
        self.target_stds = lab.std(axis=(0, 1))
        
        return self
    
    def transform(self, img: np.ndarray) -> np.ndarray:
        """
        Apply color normalization.
        
        Args:
            img: Source RGB image (H, W, 3), uint8
        
        Returns:
            Normalized RGB image (H, W, 3), uint8
        """
        import cv2
        
        if self.target_means is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        # Convert to LAB
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Compute source statistics
        src_means = lab.mean(axis=(0, 1))
        src_stds = lab.std(axis=(0, 1))
        
        # Avoid division by zero
        src_stds = np.maximum(src_stds, 1e-6)
        
        # Normalize: subtract mean, scale by std ratio, add target mean
        lab = (lab - src_means) * (self.target_stds / src_stds) + self.target_means
        
        # Clip to valid range
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        
        # Convert back to RGB
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return rgb
