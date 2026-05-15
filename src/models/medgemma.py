import gc
import logging
import time
from typing import List

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class MedGemmaModel:
    """Wrapper around MedGemma multimodal language model with 4-bit quantization."""

    def __init__(
        self,
        model_name: str = "google/medgemma-4b-it",
        use_4bit: bool = True,
        device_map: str = "auto",
        cache_dir: str = ".cache/medgemma",
    ) -> None:
        """
        Load MedGemma with optional 4-bit BitsAndBytes quantization.

        Args:
            model_name: HuggingFace model ID.
            use_4bit: Whether to apply NF4 4-bit quantization.
            device_map: Device placement strategy ('auto' for multi-GPU/CPU offload).
            cache_dir: Directory for caching model weights.
        """
        import pathlib
        pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)

        bnb_config = self._load_4bit_config() if use_4bit else None

        logger.info("Loading MedGemma model '%s' (4bit=%s)...", model_name, use_4bit)
        t0 = time.time()

        self.processor = AutoProcessor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            cache_dir=cache_dir,
        )
        self.model.eval()
        elapsed = time.time() - t0
        logger.info("MedGemma loaded in %.1fs.", elapsed)

    def __enter__(self) -> "MedGemmaModel":
        return self

    def __exit__(self, *args) -> None:
        self.clear_cache()

    def _load_4bit_config(self) -> BitsAndBytesConfig:
        """Return BitsAndBytes NF4 4-bit quantization config."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        """
        Generate text for a single image-prompt pair.

        Args:
            image: PIL image input.
            prompt: Text prompt for the model.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature; 0.0 uses greedy decoding.

        Returns:
            Generated text string.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True,
        ).to(self.model.device)

        do_sample = temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature

        try:
            with torch.inference_mode(), torch.cuda.amp.autocast():
                output_ids = self.model.generate(**inputs, **gen_kwargs)
            input_len = inputs["input_ids"].shape[1]
            generated = self.processor.decode(
                output_ids[0][input_len:], skip_special_tokens=True
            )
            return generated.strip()
        except torch.cuda.OutOfMemoryError:
            logger.error("OOM — clearing cache and retrying with max_new_tokens=128")
            self.clear_cache()
            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
            input_len = inputs["input_ids"].shape[1]
            return self.processor.decode(
                output_ids[0][input_len:], skip_special_tokens=True
            ).strip()
        except Exception as e:
            logger.error("Generation failed: %s", e)
            raise

    def generate_batch(
        self,
        items: List[dict],
        max_new_tokens: int = 512,
    ) -> List[str]:
        """
        Generate text for a list of image-prompt dicts.

        Args:
            items: List of dicts with keys 'image' (PIL.Image) and 'prompt' (str).
            max_new_tokens: Maximum tokens per generation.

        Returns:
            List of generated text strings.
        """
        results = []
        for item in tqdm(items, desc="Batch generation"):
            result = self.generate(
                image=item["image"],
                prompt=item["prompt"],
                max_new_tokens=max_new_tokens,
            )
            results.append(result)
            self.clear_cache()
        return results

    def clear_cache(self) -> None:
        """Free GPU memory and trigger garbage collection."""
        torch.cuda.empty_cache()
        gc.collect()
