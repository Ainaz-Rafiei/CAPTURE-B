import torch
import random
import numpy as np
import pandas as pd
import random
import glob
import os
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, QhullError
from transformers import AutoModel

class CaptionAnalyzer:
    def __init__(self, prism_model, device="cuda"):
        self.device = device
        self.prism_processor = prism_model  
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    def add_noise(self, tensor, noise_level=0.1):
        noise = torch.randn_like(tensor) * noise_level
        return tensor + noise

    def generate_captions(self, latents, temp_range=(0.01, 1.5), steps=30, noise_level=0.0):
        temperatures = torch.linspace(temp_range[0], temp_range[1], steps).tolist()
        captions, temps_used = [], []

        for temp in temperatures:
            with torch.autocast(self.device, torch.float16), torch.inference_mode():
                noisy_latents = latents + torch.randn_like(latents) * noise_level if noise_level > 0 else latents
                caption = self.prism_processor.generate_caption(noisy_latents)

                if isinstance(caption, list) and caption:
                    caption = caption[0]
                if isinstance(caption, str) and caption.strip():
                    captions.append(caption.strip())
                    temps_used.append(temp)

        return captions, temps_used

    def embed_captions(self, captions):
        clean_captions = [c for c in captions if isinstance(c, str) and c.strip()]
        if not clean_captions:
            raise ValueError("No valid captions to embed.")
        return self.text_encoder.encode(clean_captions)

    def reduce_embeddings_to_2d(self, caption_embeddings):
        return PCA(n_components=2).fit_transform(caption_embeddings)

    def cluster_embeddings(self, caption_pca):
        clustering = DBSCAN(eps=0.25, min_samples=3).fit(caption_pca)
        return clustering.labels_, set(clustering.labels_)

    def plot_clusters_with_convex_hulls(self, caption_pca, labels, unique_labels, temps_used):
        plt.figure(figsize=(10, 8))
        colors = plt.get_cmap("tab10")

        for label in unique_labels:
            if label == -1:
                continue

            points = caption_pca[labels == label]
            plt.scatter(points[:, 0], points[:, 1], label=f'Cluster {label}', color=colors(label % 10))

            if len(points) >= 3:
                try:
                    hull = ConvexHull(points, qhull_options='QJ')
                    for simplex in hull.simplices:
                        plt.plot(points[simplex, 0], points[simplex, 1], 'k--', linewidth=1)
                except QhullError as e:
                    print(f"[WARNING] Convex hull failed for cluster {label}: {str(e)}")

        for i, (x, y) in enumerate(caption_pca):
            plt.text(x, y, f"{temps_used[i]:.2f}", fontsize=8)

        plt.title("Caption Embedding Clusters Across Temperatures")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def get_zero_shot_predictions_with_uncertainty(self, image_embedding, image_latents, pos_prompts, neg_prompts, num_runs=10, noise_std_emb=0.01, noise_std_latents=0.5):
        predictions = []
        captions = []

        for i in range(num_runs):
            torch.manual_seed(i * 42)
            np.random.seed(i * 42)
            random.seed(i * 42)

            self.prism_processor.model.eval()
            with torch.autocast(self.device, torch.float16), torch.inference_mode():
                noisy_emb = image_embedding + torch.randn_like(image_embedding) * noise_std_emb

                scores = self.prism_processor.zero_shot_classification(
                    noisy_emb,
                    neg_prompts=neg_prompts,
                    pos_prompts=pos_prompts
                ).cpu()

                probs = torch.softmax(scores, dim=1).numpy()
                predictions.append(probs)

                noisy_latents = image_latents + torch.randn_like(image_latents) * noise_std_latents

                ids = self.prism_processor.model.generate(
                    key_value_states=noisy_latents,
                    do_sample=True,
                    temperature=1.0,  # Fixed temperature
                    num_beams=5,
                )
                caption = self.prism_processor.model.untokenize(ids)[0].strip("</s>").strip()
                captions.append(caption)

        predictions = np.array(predictions)
        if predictions.ndim == 3 and predictions.shape[1] == 1:
            predictions = predictions.squeeze(axis=1)

        uncertainty = np.var(predictions, axis=0)
        return predictions, uncertainty, captions

    def analyze_and_save_for_temperature(self, image_embedding, image_latents, pos_prompts, neg_prompts, num_runs=20, fixed_temperature=1.0, noise_std_emb=0.01, noise_std_latents=0.5, output_csv="results.csv"):

        rows = []

        for i in range(num_runs):
            torch.manual_seed(i * 42)  
            np.random.seed(i * 42)
            random.seed(i * 42)

            # Set model to evaluation mode
            self.prism_processor.model.eval()

            # Add noise to embeddings and latents
            noisy_emb = image_embedding + torch.randn_like(image_embedding) * noise_std_emb
            noisy_latents = image_latents + torch.randn_like(image_latents) * noise_std_latents

            with torch.autocast(self.device, torch.float16), torch.inference_mode():
                # Get zero-shot classification scores
                scores = self.prism_processor.zero_shot_classification(
                    noisy_emb,
                    neg_prompts=neg_prompts,
                    pos_prompts=pos_prompts
                ).cpu()

                # Calculate uncertainty
                uncertainty = abs(scores[0, 1] - scores[0, 0]).item()

                # Generate a caption with the fixed temperature
                ids = self.prism_processor.model.generate(
                    key_value_states=noisy_latents,
                    do_sample=True,
                    temperature=fixed_temperature,  # Fixed temperature
                    num_beams=1,
                )
                caption = self.prism_processor.model.untokenize(ids)[0].strip("</s>").strip()

                # Prepare the data for this run and add to the rows list
                row = {
                    "Identifier": "patient_image",  
                    "Temperature": fixed_temperature,
                    "Generated Caption": caption,
                    "Term Value": scores[0, 1].item(),  
                    "Uncertainty Value": uncertainty,
                    "Response": "Ductal" if scores[0, 1] > scores[0, 0] else "Lobular"
                }

                rows.append(row)

        df = pd.DataFrame(rows)

        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"[SAVED] Analysis results to {output_csv}")

        return df  



    def analyze_and_plot_for_temperature(self, image_embedding, image_latents, pos_prompts, neg_prompts, num_runs=20, fixed_temperature=1.0, noise_std_emb=0.01, noise_std_latents=0.5):
        captions = []
        temps_used = []

        for i in range(num_runs):
            torch.manual_seed(i * 42)
            np.random.seed(i * 42)
            random.seed(i * 42)

            self.prism_processor.model.eval()
            with torch.autocast(self.device, torch.float16), torch.inference_mode():
                noisy_emb = image_embedding + torch.randn_like(image_embedding) * noise_std_emb

                scores = self.prism_processor.zero_shot_classification(
                    noisy_emb,
                    neg_prompts=neg_prompts,
                    pos_prompts=pos_prompts
                ).cpu()

                # Generate caption with fixed temperature
                noisy_latents = image_latents + torch.randn_like(image_latents) * noise_std_latents

                ids = self.prism_processor.model.generate(
                    key_value_states=noisy_latents,
                    do_sample=True,
                    temperature=fixed_temperature,  
                    num_beams=5,
                )
                caption = self.prism_processor.model.untokenize(ids)[0].strip("</s>").strip()
                captions.append(caption)
                temps_used.append(fixed_temperature)

        # Embed captions
        embeddings = self.embed_captions(captions)

        # Reduce to 2D using PCA
        pca = self.reduce_embeddings_to_2d(embeddings)

        # Cluster the 2D points
        labels, unique_labels = self.cluster_embeddings(pca)

        # Plot the clusters with convex hulls
        self.plot_clusters_with_convex_hulls(pca, labels, unique_labels, temps_used)
