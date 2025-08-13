import os
import logging
import argparse
import json
import time
from datetime import datetime
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
            # Use LANCZOS resampling with compatibility for different PIL versions
            try:
                im = im.resize(target_size, Image.Resampling.LANCZOS)
            except AttributeError:
                # Fallback for older PIL versions
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
        
        # Check for potential issues (updated thresholds per requirements)
        has_color_cast = color_cast_ratio > 1.3  # Threshold for significant color cast
        is_overexposed = np.mean(img_float) > 220  # Average pixel value too high
        is_underexposed = np.mean(img_float) < 35  # Average pixel value too low
        
        # Additional quality checks
        severe_color_cast = color_cast_ratio > 1.5  # Severe color cast threshold
        extreme_overexposed = np.mean(img_float) > 240
        extreme_underexposed = np.mean(img_float) < 15
        
        return {
            'channel_means': {'r': float(r_mean), 'g': float(g_mean), 'b': float(b_mean)},
            'channel_stds': {'r': float(r_std), 'g': float(g_std), 'b': float(b_std)},
            'color_cast_ratio': float(color_cast_ratio),
            'has_color_cast': has_color_cast,
            'severe_color_cast': severe_color_cast,
            'is_overexposed': is_overexposed,
            'is_underexposed': is_underexposed,
            'extreme_overexposed': extreme_overexposed,
            'extreme_underexposed': extreme_underexposed,
            'overall_brightness': float(np.mean(img_float))
        }
        
    except Exception as e:
        logging.error(f"Error analyzing color distribution: {str(e)}")
        return None


def should_skip_file(image_path, save_path, force_reprocess=False):
    """
    Determine if a file should be skipped during processing.
    
    Args:
        image_path: Path to input image
        save_path: Path to processed image output
        force_reprocess: If True, process even if output exists
        
    Returns:
        tuple: (should_skip: bool, reason: str)
    """
    # Check if already processed
    if os.path.exists(save_path) and not force_reprocess:
        return True, "Already processed"
    
    # Check if input is actually from processed directory
    if "processed" in os.path.normpath(image_path):
        return True, "Input from processed directory"
    
    # Check if input is already PNG (and in raw directory)
    if image_path.lower().endswith('.png') and not force_reprocess:
        # Only skip if it's the same base name and already in processed
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        expected_output = os.path.join(os.path.dirname(save_path), base_name + ".png")
        if os.path.exists(expected_output):
            return True, "PNG already processed"
    
    return False, ""


def calculate_quality_score(color_analysis):
    """
    Calculate overall quality score based on color analysis.
    
    Args:
        color_analysis: Dict from analyze_color_distribution
        
    Returns:
        float: Quality score between 0-1 (1 = perfect quality)
    """
    if not color_analysis:
        return 0.5  # Neutral score if no analysis
    
    score = 1.0
    
    # Penalize color cast
    color_cast_ratio = color_analysis.get('color_cast_ratio', 1.0)
    if color_cast_ratio > 1.5:  # Severe color cast
        score -= 0.4
    elif color_cast_ratio > 1.3:  # Moderate color cast
        score -= 0.2
    
    # Penalize exposure issues
    if color_analysis.get('is_overexposed', False):
        score -= 0.3
    if color_analysis.get('is_underexposed', False):
        score -= 0.3
    
    # Check brightness range (ideal medical images should have good contrast)
    brightness = color_analysis.get('overall_brightness', 127)
    if brightness < 20 or brightness > 235:  # Extreme exposure
        score -= 0.2
    
    return max(0.0, min(1.0, score))


def process_with_quality_check(image_path, save_path, quality_threshold=0.8, strict_mode=False):
    """
    Process image with comprehensive quality checks.
    
    Args:
        image_path: Path to input image
        save_path: Path to save processed image  
        quality_threshold: Minimum quality score (0-1) to save image
        strict_mode: If True, don't save images with any quality issues
        
    Returns:
        dict: Processing results with keys:
            - success: bool
            - saved: bool (whether image was saved)
            - quality_score: float (0-1)
            - quality_issues: list of issue descriptions
            - color_analysis: dict with color metrics
            - output_path: str
            - error: str (if success=False)
    """
    try:
        # Process image with color analysis
        color_analysis = convert_and_standardize(image_path, save_path, analyze_colors=True)
        
        # Comprehensive quality assessment
        quality_issues = []
        
        if color_analysis:
            # Check for color cast
            color_cast_ratio = color_analysis.get('color_cast_ratio', 1.0)
            if color_cast_ratio > 1.5:
                quality_issues.append(f"Severe color cast detected (ratio: {color_cast_ratio:.2f})")
            elif color_cast_ratio > 1.3:
                quality_issues.append(f"Moderate color cast detected (ratio: {color_cast_ratio:.2f})")
            
            # Check exposure
            if color_analysis.get('is_overexposed', False):
                quality_issues.append("Image appears overexposed")
            if color_analysis.get('is_underexposed', False):
                quality_issues.append("Image appears underexposed")
            
            # Check brightness extremes
            brightness = color_analysis.get('overall_brightness', 127)
            if brightness < 20:
                quality_issues.append("Extremely dark image")
            elif brightness > 235:
                quality_issues.append("Extremely bright image")
        
        # Calculate overall quality score
        quality_score = calculate_quality_score(color_analysis)
        
        # Determine if image should be saved
        should_save = True
        if strict_mode and quality_issues:
            should_save = False
            # Remove the already saved file if it exists
            if os.path.exists(save_path):
                os.remove(save_path)
        elif quality_score < quality_threshold:
            should_save = False
            # Remove the already saved file if it exists
            if os.path.exists(save_path):
                os.remove(save_path)
        
        # Log quality issues
        if quality_issues:
            level = logging.ERROR if quality_score < 0.5 else logging.WARNING
            logging.log(level, f"Quality issues in {image_path}: {'; '.join(quality_issues)} (score: {quality_score:.2f})")
        
        return {
            'success': True,
            'saved': should_save,
            'quality_score': quality_score,
            'quality_issues': quality_issues,
            'color_analysis': color_analysis,
            'output_path': save_path
        }
        
    except Exception as e:
        return {
            'success': False,
            'saved': False,
            'quality_score': 0.0,
            'quality_issues': [],
            'color_analysis': None,
            'output_path': save_path,
            'error': str(e)
        }

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Medical image preprocessing with CLAHE enhancement and quality assessment"
    )
    parser.add_argument(
        "--force-reprocess", 
        action="store_true", 
        help="Reprocess even if output files already exist"
    )
    parser.add_argument(
        "--strict-quality", 
        action="store_true", 
        help="Skip saving images with any quality issues"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable debug-level logging"
    )
    parser.add_argument(
        "--quality-threshold", 
        type=float, 
        default=0.8, 
        help="Minimum quality score to save image (0-1, default: 0.8)"
    )
    return parser.parse_args()


def save_processing_report(report_data, report_path):
    """Save processing report as JSON."""
    try:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        logging.info(f"Processing report saved to: {report_path}")
    except Exception as e:
        logging.error(f"Failed to save processing report: {str(e)}")


def estimate_time_remaining(processed_count, total_count, elapsed_time):
    """Estimate remaining processing time."""
    if processed_count == 0:
        return "Unknown"
    
    rate = processed_count / elapsed_time
    remaining_files = total_count - processed_count
    remaining_seconds = remaining_files / rate
    
    if remaining_seconds < 60:
        return f"{remaining_seconds:.0f}s"
    elif remaining_seconds < 3600:
        return f"{remaining_seconds/60:.1f}m"
    else:
        return f"{remaining_seconds/3600:.1f}h"


# Recursively process and preserve folder structure
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    valid_exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    
    # Statistics tracking
    start_time = time.time()
    total_files = 0
    processed_files = 0
    skipped_files = 0
    failed_files = 0
    quality_issues_count = 0
    low_quality_saved = 0
    quality_rejected = 0
    
    # Detailed tracking
    failed_details = []
    quality_issues_details = []
    skipped_details = []
    
    # Count total files first for progress tracking
    total_file_count = 0
    for root, dirs, files in os.walk(src_dir):
        for file_name in files:
            if file_name.lower().endswith(valid_exts):
                total_file_count += 1
    
    logging.info(f"Starting image preprocessing with LAB-space CLAHE")
    logging.info(f"Input directory: {src_dir}")
    logging.info(f"Output directory: {out_dir}")
    logging.info(f"Target size: {target_size}")
    logging.info(f"Output format: PNG (lossless)")
    logging.info(f"Quality threshold: {args.quality_threshold}")
    logging.info(f"Strict quality mode: {args.strict_quality}")
    logging.info(f"Force reprocess: {args.force_reprocess}")
    logging.info(f"Total files to process: {total_file_count}")
    logging.info("-" * 60)
    
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
                
                # Check if file should be skipped
                should_skip, skip_reason = should_skip_file(src_path, out_path, args.force_reprocess)
                
                if should_skip:
                    skipped_files += 1
                    skipped_details.append({"file": src_path, "reason": skip_reason})
                    logging.debug(f"Skipping [{total_files}/{total_file_count}]: {src_path} ({skip_reason})")
                    continue
                
                try:
                    # Progress indicator with time estimate
                    elapsed = time.time() - start_time
                    time_est = estimate_time_remaining(processed_files + failed_files, 
                                                     total_file_count - skipped_files, elapsed) if elapsed > 5 else "Calculating..."
                    
                    logging.info(f"Processing [{total_files}/{total_file_count}]: {file_name} (ETA: {time_est})")
                    
                    # Use enhanced processing with quality checks
                    result = process_with_quality_check(
                        src_path, out_path, 
                        quality_threshold=args.quality_threshold,
                        strict_mode=args.strict_quality
                    )
                    
                    if result['success']:
                        if result['saved']:
                            processed_files += 1
                            logging.info(f"✓ Successfully processed: {base_name}.png (quality: {result['quality_score']:.2f})")
                        else:
                            quality_rejected += 1
                            logging.warning(f"✗ Quality rejected: {base_name}.png (score: {result['quality_score']:.2f})")
                        
                        # Track quality issues
                        if result.get('quality_issues'):
                            quality_issues_count += 1
                            quality_issues_details.append({
                                "file": src_path,
                                "issues": result['quality_issues'],
                                "score": result['quality_score'],
                                "saved": result['saved']
                            })
                            
                            if result['saved'] and result['quality_score'] < args.quality_threshold:
                                low_quality_saved += 1
                                
                    else:
                        failed_files += 1
                        error_msg = result.get('error', 'Unknown error')
                        failed_details.append({"file": src_path, "error": error_msg})
                        logging.error(f"✗ Failed to process {file_name}: {error_msg}")
                        
                except Exception as e:
                    failed_files += 1
                    error_msg = str(e)
                    failed_details.append({"file": src_path, "error": error_msg})
                    logging.error(f"✗ Failed to process {file_name}: {error_msg}")
    
    # Calculate final statistics
    end_time = time.time()
    total_time = end_time - start_time
    
    # Create comprehensive report
    report_data = {
        "processing_summary": {
            "timestamp": datetime.now().isoformat(),
            "total_processing_time_seconds": total_time,
            "parameters": {
                "force_reprocess": args.force_reprocess,
                "strict_quality": args.strict_quality,
                "quality_threshold": args.quality_threshold,
                "target_size": target_size,
                "source_directory": src_dir,
                "output_directory": out_dir
            }
        },
        "statistics": {
            "total_files_found": total_files,
            "files_processed_successfully": processed_files,
            "files_skipped": skipped_files,
            "files_failed": failed_files,
            "files_with_quality_issues": quality_issues_count,
            "low_quality_but_saved": low_quality_saved,
            "quality_rejected": quality_rejected,
            "success_rate_percent": (processed_files / max(total_files - skipped_files, 1)) * 100
        },
        "details": {
            "failed_files": failed_details,
            "quality_issues": quality_issues_details,
            "skipped_files": skipped_details
        }
    }
    
    # Save report
    report_path = os.path.join(out_dir, "preprocessing_report.json")
    save_processing_report(report_data, report_path)
    
    # Final statistics display
    logging.info("=" * 60)
    logging.info("PREPROCESSING COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Total processing time: {total_time:.1f} seconds")
    logging.info(f"Total files found: {total_files}")
    logging.info(f"Files processed successfully: {processed_files}")
    logging.info(f"Files skipped (already processed): {skipped_files}")
    logging.info(f"Files failed: {failed_files}")
    logging.info(f"Files with quality issues: {quality_issues_count}")
    if args.strict_quality or args.quality_threshold > 0:
        logging.info(f"Quality rejected files: {quality_rejected}")
        logging.info(f"Low quality but saved: {low_quality_saved}")
    logging.info(f"Success rate: {(processed_files / max(total_files - skipped_files, 1)) * 100:.1f}%")
    logging.info(f"Processing report: {report_path}")
    logging.info("=" * 60)
                    