"""
Handwritten Notes OCR - Convert handwritten images to text using TrOCR.
"""

import os
from pathlib import Path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image


class HandwrittenOCR:
    """OCR processor for handwritten notes using Microsoft TrOCR."""
    
    def __init__(self, model_name: str = "microsoft/trocr-large-handwritten"):
        """Initialize the OCR model and processor."""
        print(f"Loading model: {model_name}...")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        print("Model loaded successfully!")
    
    def process_image(self, image_path: str) -> str:
        """
        Process a single image and return the extracted text.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Extracted text from the image.
        """
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text
    
    def process_directory(self, input_dir: str, output_dir: str) -> dict:
        """
        Process all images in a directory and save results.
        
        Args:
            input_dir: Directory containing input images.
            output_dir: Directory to save text output files.
            
        Returns:
            Dictionary mapping image filenames to extracted text.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        
        for image_file in input_path.iterdir():
            if image_file.suffix.lower() in image_extensions:
                print(f"Processing: {image_file.name}")
                text = self.process_image(str(image_file))
                results[image_file.name] = text
                
                # Save text to output file
                output_file = output_path / f"{image_file.stem}.txt"
                with open(output_file, 'w') as f:
                    f.write(text)
                print(f"  -> Saved to: {output_file.name}")
        
        return results


def main():
    """Main entry point for processing handwritten notes."""
    # Default paths
    base_dir = Path(__file__).parent
    input_dir = base_dir / "data" / "input"
    output_dir = base_dir / "data" / "output"
    
    # Initialize OCR
    ocr = HandwrittenOCR()
    
    # Process all images
    results = ocr.process_directory(str(input_dir), str(output_dir))
    
    print(f"\nProcessed {len(results)} image(s)")
    for filename, text in results.items():
        print(f"\n--- {filename} ---")
        print(text)


if __name__ == "__main__":
    main()