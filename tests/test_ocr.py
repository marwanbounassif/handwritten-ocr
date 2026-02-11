"""
End-to-end tests for the Handwritten OCR pipeline.
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from create_notes import HandwrittenOCR


def create_test_image(text: str, output_path: str):
    """Create a simple test image with text."""
    # Create a white image
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    
    # Use default font
    draw.text((10, 30), text, fill='black')
    
    img.save(output_path)
    print(f"Created test image: {output_path}")


def test_single_image():
    """Test processing a single image."""
    print("\n=== Test: Single Image Processing ===")
    
    # Create test image
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    
    test_image = test_dir / "test_hello.png"
    create_test_image("Hello World", str(test_image))
    
    # Process with OCR
    ocr = HandwrittenOCR()
    result = ocr.process_image(str(test_image))
    
    print(f"Input image: {test_image}")
    print(f"OCR output: {result}")
    print("✓ Single image test completed")
    
    return result


def test_directory_processing():
    """Test processing a directory of images."""
    print("\n=== Test: Directory Processing ===")
    
    test_dir = Path(__file__).parent / "test_data"
    output_dir = Path(__file__).parent / "test_output"
    
    test_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Create multiple test images
    test_texts = ["Sample text one", "Sample text two", "Sample text three"]
    for i, text in enumerate(test_texts):
        create_test_image(text, str(test_dir / f"sample_{i}.png"))
    
    # Process directory
    ocr = HandwrittenOCR()
    results = ocr.process_directory(str(test_dir), str(output_dir))
    
    print(f"\nProcessed {len(results)} images")
    for filename, text in results.items():
        print(f"  {filename}: {text}")
    
    # Check output files exist
    for txt_file in output_dir.glob("*.txt"):
        print(f"  Output file: {txt_file.name}")
    
    print("✓ Directory processing test completed")
    
    return results


def test_model_loading():
    """Test that model loads correctly."""
    print("\n=== Test: Model Loading ===")
    
    ocr = HandwrittenOCR()
    
    assert ocr.processor is not None, "Processor failed to load"
    assert ocr.model is not None, "Model failed to load"
    
    print("✓ Model loading test passed")


def cleanup_test_files():
    """Clean up test files."""
    import shutil
    
    test_dir = Path(__file__).parent / "test_data"
    output_dir = Path(__file__).parent / "test_output"
    
    if test_dir.exists():
        shutil.rmtree(test_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    print("\n✓ Cleaned up test files")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Handwritten OCR - End-to-End Tests")
    print("=" * 50)
    
    try:
        test_model_loading()
        test_single_image()
        test_directory_processing()
        
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        raise
    finally:
        cleanup_test_files()


if __name__ == "__main__":
    run_all_tests()
