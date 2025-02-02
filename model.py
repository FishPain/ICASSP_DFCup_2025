import torch
from torch import nn
from torchvision import models
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import clip
from PIL import Image

import random
# Check interpolation compatibility
try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# Constants
MAX_NEW_TOKENS = 96
DEFAULT_DEVICE = "cuda"

class QwenExtractor(nn.Module):
    def __init__(self, model_type="Qwen/Qwen2-VL-2B-Instruct", device=DEFAULT_DEVICE):
        super().__init__()
        self.device = device
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_type,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_type)

        # Define different types of analyses as a list
        self.analysis_types = [
            "*textures*",
            "*lighting*",
            "*facial features*",
            "*background consistency*",
            "*noticeable artifacts like mismatched reflections*",
            "*unnatural edges*",
            "*color mismatches*",
            "*shadow irregularities*",
            "*blur inconsistencies*"
        ]

        # Base prompt for reconstruction
        self.base_prompt = "Analyze the image focusing on the following aspects: {}. *Always start with 'This image is <Description>' and repeat the pattern.*"

    def build_message(self, image_dir_list, random_prompt=False):
        """Creates a system and user message for each image."""
        system_msg = {
            "role": "system",
            "content": "Before making any decisions, always address the user's requirements.",
        }

        if random_prompt:
            # Randomly select a subset of analysis types
            selected_analyses = random.sample(
                self.analysis_types, k=random.randint(2, len(self.analysis_types))
            )
            # Construct the prompt with the selected analyses
            selected_prompt = self.base_prompt.format(", ".join(selected_analyses))
        else:
            # Default prompt with all analyses
            selected_prompt = self.base_prompt.format(", ".join(self.analysis_types))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": selected_prompt},
                ],
            }
            for img in image_dir_list
        ]

        return [system_msg] + messages


    def decode(self, inputs, generated_ids):
        """Decodes the model's output."""
        generated_ids = generated_ids[1:]
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def get_preloaded_embeddings(self, image_dir_list, loaded_embeddings):
        """Loads precomputed embeddings if available."""
        return [
            torch.tensor(loaded_embeddings[img], device=self.device)
            for img in image_dir_list
        ]

    def forward(
        self, image_dir_list, epoch, loaded_embeddings, use_custom_embeddings=False
    ):
        messages = self.build_message(image_dir_list, random_prompt=False)
        texts = [
            self.processor.apply_chat_template(
                [msg], tokenize=False, add_generation_prompt=True
            )
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        if epoch > 0 or use_custom_embeddings:
            preloaded_embeddings = self.get_preloaded_embeddings(
                image_dir_list, loaded_embeddings
            )
            return self.decode(inputs, preloaded_embeddings), image_inputs

        self.model.eval()
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
            for img, embedding in zip(image_dir_list, generated_ids.cpu().numpy()):
                loaded_embeddings[img] = embedding

        return self.decode(inputs, generated_ids), image_inputs


class CLIPExtractor(nn.Module):
    def __init__(self, model_type="ViT-B/32", device=DEFAULT_DEVICE, download_dir=None):
        super().__init__()
        self.device = device
        self.model, self.preprocess = clip.load(
            model_type, device=device, download_root=download_dir
        )

    def forward(self, prompt, images):
        with torch.no_grad():
            text_tokens = clip.tokenize(prompt, truncate=True).to(self.device)
            processed_imgs = torch.stack([self.preprocess(img) for img in images]).to(
                self.device
            )
            image_features = self.model.encode_image(processed_imgs)
            text_features = self.model.encode_text(text_tokens)

            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features, processed_imgs


class ResnetModel(nn.Module):
    def __init__(self, device=DEFAULT_DEVICE):
        super().__init__()
        self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 512)

    def forward(self, x):
        return self.model(x)


class CustomClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 512 * 3  # Combined embedding sizes
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.4)

    def forward(self, bb_embed, img_emb, text_emb):
        x = torch.cat([bb_embed, img_emb, text_emb], dim=1)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        return self.fc3(x)
