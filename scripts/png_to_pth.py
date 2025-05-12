import os
import torch
import glob
from utils import TileExtractor
from PIL import Image


def png_to_pth(input_dir, output_dir, device="cpu"):
    tile_extractor = TileExtractor(device=device)
    
    os.makedirs(output_dir, exist_ok=True)

    png_files = glob.glob(os.path.join(input_dir, "*.png"))
    print(f"Found {len(png_files)} PNG files in {input_dir}")


    for i, png_path in enumerate(png_files, 1):
        try:
            image = Image.open(png_path).convert("RGB")
            embeddings = tile_extractor.extract_embeddings(image)
          
            save_path = os.path.join(output_dir, f"image_{i}.pth")
            tile_extractor.save_embeddings(embeddings, save_path)

        except Exception as e:
            print(f"Error processing {png_path}: {e}")

# input_directory = "/Data/img-output"
# output_directory = "/Data/embeddings-output"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# png_to_pth(input_directory, output_directory, device=device)
