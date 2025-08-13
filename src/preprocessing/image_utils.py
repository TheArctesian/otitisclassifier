import os
import logging
from PIL import Image
import numpy as np
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

src_dir = "data/raw"
out_dir = "data/processed"
target_size = (500, 500)  # Change this as needed

def convert_and_standardize(image_path, save_path, analyze_colors=False):
    """
    Convert and preprocess medical images using LAB color space CLAHE.
    
    Args:
        image_path: Path to input image
        save_path: Path to save processed image
        analyze_colors: Whether to perform color distribution analysis
        
    Returns:
        dict: Color analysis results if analyze_colors=True, else None
    """
    try:
        # Load and validate image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")
            
        with Image.open(image_path) as im:
            # Log original image properties
            logging.debug(f"Processing {image_path}: {im.format}, {im.mode}, {im.size}")
            
            im = im.convert('RGB')
            im = im.resize(target_size, Image.LANCZOS)
            img_array = np.array(im).astype(np.uint8)
        
        # Validate image array
        if img_array.size == 0:
            raise ValueError(f"Empty image array for {image_path}")
            
        # Apply CLAHE in LAB color space (only to L channel)
        lab_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        
        # Create CLAHE object with medical-appropriate parameters
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Apply CLAHE only to the L (lightness) channel
        lab_img[:, :, 0] = clahe.apply(lab_img[:, :, 0])
        
        # Convert back to RGB
        clahe_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
        
        # Validate processed image
        if clahe_img.shape != img_array.shape:
            raise ValueError(f"Shape mismatch after CLAHE processing: {clahe_img.shape} vs {img_array.shape}")
        
        # Perform color analysis if requested
        color_analysis = None
        if analyze_colors:
            color_analysis = analyze_color_distribution(clahe_img)
            if color_analysis and color_analysis.get('has_color_cast', False):
                logging.warning(f"Color cast detected in {image_path}: ratio={color_analysis.get('color_cast_ratio', 'N/A')}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save in PNG format for lossless compression (critical for medical imaging)
        img_out = Image.fromarray(clahe_img)
        img_out.save(save_path, "PNG", optimize=True)
        
        # Verify saved image
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Failed to save processed image: {save_path}")
            
        logging.debug(f"Successfully saved processed image: {save_path}")
        
        return color_analysis
        
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        raise


def analyze_color_distribution(img_array):
    """
    Analyze color distribution to detect color casts or imaging issues.
    
    Args:
        img_array: RGB image array (numpy array)
        
    Returns:
        dict: Color analysis metrics including channel means, color cast detection
    """
    try:
        # Convert to float for calculations
        img_float = img_array.astype(np.float32)
        
        # Calculate channel statistics
        r_mean = np.mean(img_float[:, :, 0])
        g_mean = np.mean(img_float[:, :, 1])
        b_mean = np.mean(img_float[:, :, 2])
        
        # Calculate channel standard deviations
        r_std = np.std(img_float[:, :, 0])
        g_std = np.std(img_float[:, :, 1])
        b_std = np.std(img_float[:, :, 2])
        
        # Detect color cast (significant imbalance between channels)
        mean_channels = [r_mean, g_mean, b_mean]
        max_mean = max(mean_channels)
        min_mean = min(mean_channels)
        color_cast_ratio = max_mean / (min_mean + 1e-6)
        
        # Check for potential issues
        has_color_cast = color_cast_ratio > 1.3  # Threshold for significant color cast
        is_overexposed = np.mean(img_float) > 220  # Average pixel value too high
        is_underexposed = np.mean(img_float) < 35  # Average pixel value too low
        
        return {
            'channel_means': {'r': float(r_mean), 'g': float(g_mean), 'b': float(b_mean)},
            'channel_stds': {'r': float(r_std), 'g': float(g_std), 'b': float(b_std)},
            'color_cast_ratio': float(color_cast_ratio),
            'has_color_cast': has_color_cast,
            'is_overexposed': is_overexposed,
            'is_underexposed': is_underexposed,
            'overall_brightness': float(np.mean(img_float))
        }
        
    except Exception as e:
        logging.error(f"Error analyzing color distribution: {str(e)}")
        return None


def process_with_quality_check(image_path, save_path, quality_threshold=0.8):
    """
    Process image with quality checks and color analysis.
    
    Args:
        image_path: Path to input image
        save_path: Path to save processed image
        quality_threshold: Threshold for quality checks
        
    Returns:
        dict: Processing results including quality metrics
    """
    try:
        # Process image with color analysis
        color_analysis = convert_and_standardize(image_path, save_path, analyze_colors=True)
        
        # Quality assessment
        quality_issues = []
        
        if color_analysis:
            if color_analysis.get('has_color_cast', False):
                quality_issues.append(f"Color cast detected (ratio: {color_analysis.get('color_cast_ratio', 'N/A'):.2f})")
            
            if color_analysis.get('is_overexposed', False):
                quality_issues.append("Image appears overexposed")
                
            if color_analysis.get('is_underexposed', False):
                quality_issues.append("Image appears underexposed")
        
        # Log quality issues
        if quality_issues:
            logging.warning(f"Quality issues in {image_path}: {'; '.join(quality_issues)}")
        
        return {
            'success': True,
            'color_analysis': color_analysis,
            'quality_issues': quality_issues,
            'output_path': save_path
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'output_path': save_path
        }

# Recursively process and preserve folder structure
if __name__ == "__main__":
    valid_exts = (".tif", ".tiff", ".png", ".jpg")
    
    # Statistics tracking
    total_files = 0
    processed_files = 0
    skipped_files = 0
    failed_files = 0
    quality_issues_count = 0
    
    logging.info(f"Starting image preprocessing with LAB-space CLAHE")
    logging.info(f"Input directory: {src_dir}")
    logging.info(f"Output directory: {out_dir}")
    logging.info(f"Target size: {target_size}")
    logging.info(f"Output format: PNG (lossless)")
    
    for root, dirs, files in os.walk(src_dir):
        rel_path = os.path.relpath(root, src_dir)
        target_dir = os.path.join(out_dir, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        for file_name in files:
            if file_name.lower().endswith(valid_exts):
                total_files += 1
                src_path = os.path.join(root, file_name)
                base_name = os.path.splitext(file_name)[0]
                out_path = os.path.join(target_dir, base_name + ".png")
                
                # Process ONLY if the output file does NOT already exist
                if not os.path.exists(out_path):
                    try:
                        logging.info(f"Processing [{total_files}]: {src_path}")
                        
                        # Use enhanced processing with quality checks
                        result = process_with_quality_check(src_path, out_path)
                        
                        if result['success']:
                            processed_files += 1
                            logging.info(f"Successfully processed: {out_path}")
                            
                            # Track quality issues
                            if result.get('quality_issues'):
                                quality_issues_count += 1
                                
                        else:
                            failed_files += 1
                            logging.error(f"Failed to process {src_path}: {result.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        failed_files += 1
                        logging.error(f"Failed to process {src_path}: {str(e)}")
                else:
                    skipped_files += 1
                    logging.debug(f"Skipping existing file: {out_path}")
    
    # Final statistics
    logging.info("=" * 60)
    logging.info("PREPROCESSING COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Total files found: {total_files}")
    logging.info(f"Files processed: {processed_files}")
    logging.info(f"Files skipped (already exist): {skipped_files}")
    logging.info(f"Files failed: {failed_files}")
    logging.info(f"Files with quality issues: {quality_issues_count}")
    logging.info(f"Success rate: {(processed_files / max(total_files - skipped_files, 1)) * 100:.1f}%")
    logging.info("=" * 60)
                    