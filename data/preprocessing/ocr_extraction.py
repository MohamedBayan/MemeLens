"""
@author: Firoj Alam
Modified: 14th April, 2023
"""

# importing the libraries
import pandas as pd
import shutil
import json
from PIL import Image
import re
import argparse
import easyocr
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
def create_batches(image_list, batch_size=32):
    """Split the image list into batches of a specified size."""
    for i in range(0, len(image_list), batch_size):
        yield image_list[i:i + batch_size]

def get_text(data):
    # Extracting text from each item in the list
    extracted_texts = [item[1].strip() for item in data]
    text = '\n'.join(extracted_texts)
    return text.strip()

def process_image(image_data, out_file, reader, num_workers):
    """
    Process an image to extract text using OCR.

    :param image_data: list, list of tuples (image_path, original_json_obj)
    :param out_file: file handle for output
    :param reader: easyocr.Reader, an initialized easyocr reader instance
    :param num_workers: int, number of workers
    """
    import os
    from PIL import Image
    
    # Filter out non-existent or corrupted images
    valid_images = []
    for img_path, original_data in image_data:
        if not os.path.exists(img_path):
            print(f"Warning: Image file does not exist: {img_path}")
            continue
        
        # Check if image can be opened
        try:
            with Image.open(img_path) as img:
                img.verify()  # Verify that it's a valid image
            valid_images.append((img_path, original_data))
        except Exception as e:
            print(f"Warning: Corrupted or invalid image file: {img_path} - {str(e)}")
            continue
    
    print(f"Processing {len(valid_images)} valid images out of {len(image_data)} total images")
    
    batch_size = 8  # Reduced batch size for memory management
    batches = list(create_batches(valid_images, batch_size))
    try:
        # etext = reader.readtext(image_path, detail=0, paragraph=True, workers=num_workers)
        for batch in batches:
            print(f"Processing batch of {len(batch)} images...")
            try:
                # Extract just the image paths for batch processing
                batch_paths = [item[0] for item in batch]
                result_agg = reader.readtext_batched(batch_paths, n_width=800, n_height=600, batch_size=batch_size, paragraph=True, workers=num_workers)
                for item, (img_path, original_data) in zip(result_agg, batch):
                    # Preserve all original columns and add the text
                    json_obj = original_data.copy()
                    json_obj["text"] = get_text(item)
                    json_obj = json.dumps(json_obj, ensure_ascii=False)
                    out_file.write(json_obj + "\n")
                    out_file.flush()  # Ensure data is written immediately
            except Exception as batch_e:
                print(f"Error processing batch: {batch_e}")
                # Process images individually if batch fails
                for img_path, original_data in batch:
                    try:
                        result = reader.readtext(img_path, paragraph=True, workers=min(num_workers, 2))
                        # Preserve all original columns and add the text
                        json_obj = original_data.copy()
                        json_obj["text"] = get_text(result)
                        json_obj = json.dumps(json_obj, ensure_ascii=False)
                        out_file.write(json_obj + "\n")
                        out_file.flush()
                    except Exception as img_e:
                        print(f"Error processing individual image {img_path}: {img_e}")
                        # Write empty result for failed images but preserve original data
                        json_obj = original_data.copy()
                        json_obj["text"] = ""
                        json_obj = json.dumps(json_obj, ensure_ascii=False)
                        out_file.write(json_obj + "\n")
                        out_file.flush()

    except Exception as e:
        # print("Error in image: {}".format(image_path))
        print(f"Critical error in image processing: {e}")
        return None
    out_file.close()

def main(args):
    """
    Main function that executes the script's logic.

    :param args: argparse.Namespace, parsed command-line arguments
    """
    input_file = args.input_file
    out_file_path = args.output_file
    num_workers = args.num_workers
    lang = args.language

    # Ensure EasyOCR model directory exists
    import os
    easyocr_dir = os.path.expanduser("~/.EasyOCR")
    model_dir = os.path.join(easyocr_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize easyocr reader with specified language and English support (only once)
    print(f"Initializing EasyOCR with {lang} and English support (GPU mode)...")
    try:
        reader = easyocr.Reader([lang, 'en'], gpu=True, verbose=False)
        print("EasyOCR initialization successful!")
    except Exception as e:
        print(f"Error initializing EasyOCR: {e}")
        print("Trying CPU mode...")
        reader = easyocr.Reader([lang, 'en'], gpu=False, verbose=False)
        print("EasyOCR initialization successful (CPU mode)!")

    # Read image paths and preserve original data
    with open(input_file, 'r') as f:
        image_data = []
        for line in f:
            json_obj = json.loads(line)
            image_path = json_obj.get('img_path') or json_obj.get('image_path')
            if image_path:
                image_data.append((image_path, json_obj))
        
    print(f"Found {len(image_data)} images to process")

    out_file = open(out_file_path, 'w', encoding='utf-8')
    process_image(image_data, out_file, reader, num_workers)

    # # Process images in parallel
    # with ThreadPoolExecutor(max_workers=num_workers) as executor:
    #     futures = [executor.submit(process_image, img_path, reader, 2) for img_path in image_paths]
    #

        # for item in items:
        #     # result = future.result()
        #     # if result:
        #     out_file.write(item + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR text from image.")
    parser.add_argument("-i", "--input-file", type=str, required=True, default="input.jsonl", help="Input file containing image paths")
    parser.add_argument("-o", "--output-file", type=str, required=True, default="output.jsonl", help="Output JSONL file containing JSON objects with image paths and OCR text")
    parser.add_argument("-w", "--num-workers", type=int, required=False, default=4, help="Number of parallel workers for processing images")
    parser.add_argument("-l", "--language", type=str, required=False, default="bn", help="Language code for OCR (default: bn for Bengali)")

    args = parser.parse_args()
    main(args)

