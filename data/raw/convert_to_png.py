from PIL import Image
import os


def jpg_to_png(input_path, output_path=None):
    """
    Convert a JPG image file to PNG format.

    Args:
        input_path (str): Path to the input JPG file.
        output_path (str, optional): Path for the output PNG file.
                                    If None, uses the same directory and filename with .png extension.

    Returns:
        str: Path to the converted PNG file.
    """
    # Create output path if not specified
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".png"

    try:
        # Open the image and convert
        with Image.open(input_path) as img:
            # Ensure RGB mode (in case of CMYK JPGs)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Save as PNG
            img.save(output_path, "PNG")
            print(f"Converted: {input_path} → {output_path}")
            return output_path
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return None


def tiff_to_png(input_path, output_path=None):
    """
    Convert a TIFF image file to PNG format.

    Args:
        input_path (str): Path to the input TIFF file.
        output_path (str, optional): Path for the output PNG file.
                                    If None, uses the same directory and filename with .png extension.

    Returns:
        str: Path to the converted PNG file.
    """
    # Create output path if not specified
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".png"

    try:
        # Open the image and convert
        with Image.open(input_path) as img:
            # Handle multipage TIFFs (only convert the first page)
            if hasattr(img, "n_frames") and img.n_frames > 1:
                img.seek(0)

            # Convert to RGB if not already
            if img.mode not in ["RGB", "RGBA"]:
                img = img.convert("RGB")

            # Save as PNG
            img.save(output_path, "PNG")
            print(f"Converted: {input_path} → {output_path}")
            return output_path
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return None
