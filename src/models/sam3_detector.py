"""SAM3-backed detector utilities for the D2 attention stage."""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image


class SAM3Detector:
    """Thin wrapper around facebookresearch/sam3 inference utilities."""

    def __init__(
        self,
        class_names: Sequence[str],
        checkpoint_path: str | None = None,
        device: str = "cuda",
        load_from_hf: bool = True,
    ) -> None:
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as exc:  # pragma: no cover - guidance for setup
            raise ImportError(
                "facebookresearch/sam3 is not installed. Follow https://github.com/facebookresearch/sam3"
            ) from exc

        self.class_names = list(class_names)
        self.prompts = [self._make_prompt(name) for name in self.class_names]
        self.device = device
        self.model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            device=device,
            eval_mode=True,
            load_from_HF=load_from_hf,
            enable_segmentation=True,
        )
        self.processor = Sam3Processor(self.model)

    def _make_prompt(self, class_name: str) -> str:
        return f"a photo of {class_name}"

    @torch.inference_mode()
    def predict(self, image: np.ndarray, prompts: Iterable[str] | None = None):
        """Run SAM3 on a single RGB image and return per-class detections."""

        if prompts is None:
            prompts = self.prompts

        pil_img = Image.fromarray(image)
        inference_state = self.processor.set_image(pil_img)

        detections = []
        for prompt in prompts:
            output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
            boxes = output["boxes"].astype(np.float32)
            scores = output["scores"].astype(np.float32)
            detections.append({"prompt": prompt, "boxes": boxes, "scores": scores})

        return {"detections": detections}

    # Training hooks will be implemented once SAM3 fine-tuning is wired in PipelineRunner.
