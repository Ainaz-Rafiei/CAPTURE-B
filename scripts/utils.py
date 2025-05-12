import torch
import torchvision.transforms as T
import timm
from PIL import Image
import os
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from transformers import AutoModel
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, QhullError
import matplotlib.pyplot as plt
import random


class TileExtractor:
    def __init__(self, model_name="vit_base_patch16_224", device="cpu"):
        # Initialize the ViT model and related components
        self.device = device
        self.vit, self.projector, self.transform = self.initialize_vit(model_name, self.device)

    def initialize_vit(self, model_name, device):
        vit = timm.create_model(model_name, pretrained=True)
        vit.head = torch.nn.Identity()  # Remove classification head
        vit = vit.to(device).eval()

        # Projector to map the 768-dim features to 2560-dim
        projector = torch.nn.Linear(768, 2560).to(device)

        # Image transformation for ViT input
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

        return vit, projector, transform

    def tile_image(self, image, tile_size=224, stride=224):
        width, height = image.size
        tiles = []
        for y in range(0, height - tile_size + 1, stride):
            for x in range(0, width - tile_size + 1, stride):
                tile = image.crop((x, y, x + tile_size, y + tile_size))
                tiles.append(tile)
        return tiles

    def extract_embeddings(self, image):
        # Tile the image into patches
        tiles = self.tile_image(image)
        embeddings = []
        with torch.no_grad():
            for tile in tiles:
                img_tensor = self.transform(tile).unsqueeze(0).to(self.device)
                feat = self.vit(img_tensor)  # [1, 768]
                feat_proj = self.projector(feat)  # [1, 2560]
                embeddings.append(feat_proj.squeeze(0).cpu())

        tile_embeddings = torch.stack(embeddings)  # [num_tiles, 2560]
        return tile_embeddings

    def save_embeddings(self, embeddings, output_path):
        torch.save({'embeddings': embeddings}, output_path)
        print(f"[SAVED] -> {output_path}")

    def process_image(self, image_path, output_dir, output_filename):
        try:
            image = Image.open(image_path).convert("RGB")
            embeddings = self.extract_embeddings(image)
            save_path = os.path.join(output_dir, output_filename)
            self.save_embeddings(embeddings, save_path)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")


class VirchowTileEmbeddingExtractor:
    def __init__(self, device="cpu"):
        # Load the pre-trained Virchow model
        self.device = device
        self.model = timm.create_model(
            "hf-hub:paige-ai/Virchow", 
            pretrained=True, 
            mlp_layer=SwiGLUPacked, 
            act_layer=torch.nn.SiLU
        ).to(self.device).eval()
        
        # Resolve data configuration and transform
        self.transforms = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))

    def extract_embedding(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image)
        
        # Extract the class token and patch tokens, and concatenate them
        class_token = output[:, 0]
        patch_tokens = output[:, 1:]
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        
        return embedding

    def save_embedding(self, embedding, save_path):
        torch.save({'embedding': embedding}, save_path)
        print(f"Embedding saved to: {save_path}")

    def extract_and_save_embedding(self, image_path, save_path):
        embedding = self.extract_embedding(image_path)
        self.save_embedding(embedding, save_path)
        
    def load_embedding(self, save_path):
        loaded_embedding = torch.load(save_path)
        embedding_tensor = loaded_embedding['embedding']
        return embedding_tensor

class PrismProcessor:
    def __init__(self, model_name="paige-ai/Prism", device="cuda"):
        self.device = device
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model = self.model.to(self.device)

    def load_tile_embeddings(self, tile_sample_path):
 
        embedding_data = torch.load(tile_sample_path)
        print(f"Loaded embedding data keys: {embedding_data.keys()}")
        tile_embeddings = embedding_data['embedding'].unsqueeze(0).to(self.device)
        print(f"Tile embeddings shape: {tile_embeddings.shape}")
        return tile_embeddings

    def extract_slide_representations(self, tile_embeddings):

        with torch.autocast(self.device, torch.float16), torch.inference_mode():
            reprs = self.model.slide_representations(tile_embeddings)
        print(f"Slide image embedding shape: {reprs['image_embedding'].shape}")
        print(f"Slide image latents shape: {reprs['image_latents'].shape}")
        return reprs

    def zero_shot_classification(self, image_embedding, neg_prompts, pos_prompts):
        with torch.autocast(self.device, torch.float16), torch.inference_mode():
            scores = self.model.zero_shot(image_embedding, neg_prompts=neg_prompts, pos_prompts=pos_prompts)
        print(f"Zero-shot classification scores: {scores}")
        return scores

    def generate_caption(self, image_latents):

        with torch.autocast(self.device, torch.float16), torch.inference_mode():
            genned_ids = self.model.generate(
                key_value_states=image_latents,
                do_sample=False,
                num_beams=5,
                num_beam_groups=1,
            )
            genned_caption = self.model.untokenize(genned_ids)
        print(f"Generated caption: {genned_caption}")
        return genned_caption

    def make_prediction(self, caption, tile_embeddings):
        caption_tokens = self.model.tokenize([caption]).to(self.device)
        with torch.autocast(self.device, torch.float16), torch.inference_mode():
            output = self.model(input_ids=caption_tokens, tile_embeddings=tile_embeddings)
        print(f"Model output keys: {output.keys()}")
        return output
    def extract_slide_representations_from_merged_tiles(self, merged_tile_embeddings):
 
        with torch.autocast(self.device, torch.float16), torch.inference_mode():
            return self.model.slide_representations(merged_tile_embeddings)

    def zero_shot_classification_merged(self, image_embedding, neg_prompts, pos_prompts):

        with torch.autocast(self.device, torch.float16), torch.inference_mode():
            scores = self.model.zero_shot(image_embedding, neg_prompts=neg_prompts, pos_prompts=pos_prompts)
        return scores

    def generate_caption_for_merged_tiles(self, image_latents):
 
        with torch.autocast(self.device, torch.float16), torch.inference_mode():
            ids = self.model.generate(
                key_value_states=image_latents,
                do_sample=False,
                num_beams=5,
                num_beam_groups=1,
            )
            return self.model.untokenize(ids)[0]