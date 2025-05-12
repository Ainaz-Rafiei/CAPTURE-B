import os
import pandas as pd
from openslide import OpenSlide
from PIL import Image
import numpy as np
import torch
from utils import TileExtractor
from PIL import Image

def convert_svs_to_pth(slide_path, metadata_df, output_dir):
    tile_extractor = TileExtractor(device="cuda" if torch.cuda.is_available() else "cpu")
    
    slide_filename = os.path.basename(slide_path)
    case_id = "-".join(slide_filename.split("-")[:3])

    matched = metadata_df[metadata_df["cases.submitter_id"] == case_id]
    if not matched.empty:
        tumor_stage = matched["diagnoses.ajcc_pathologic_stage"].values[0]
        age = matched["demographic.age_at_index"].values[0]
    else:
        tumor_stage = "unknown"
        age = "unknown"

    slide = OpenSlide(slide_path)
    width, height = slide.dimensions

    output_filename = f"{case_id}_stage-{tumor_stage}_age-{age}.pth"
    output_path = os.path.join(output_dir, output_filename)

    tile_size = 256
    tiles = []
    for x in range(0, width, tile_size):
        for y in range(0, height, tile_size):
            tile = slide.read_region((x, y), 0, (tile_size, tile_size))
            tiles.append(tile.convert("RGB"))

    embeddings = tile_extractor.extract_embeddings(tiles)
    tile_extractor.save_embeddings(embeddings, output_path)

#!pip install openslide-python pillow
# metadata_file_path = "/Data/Clinical_Data/clinical.tsv"
# metadata_df = pd.read_csv(metadata_file_path, sep='\t', low_memory=False)

# slide_path = "/Data/tcga-svs/TCGA-B6-A0WZ-01A-01-BS1.c88de28c-ac1d-43b3-be4b-4f8b51ca5d68.svs"
# output_dir = "/output/svstTotiles"
# convert_svs_to_pth(slide_path, metadata_df, output_dir)
