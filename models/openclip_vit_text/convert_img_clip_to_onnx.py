from PIL import Image
import torch
import io
import base64
from io import BytesIO

import open_clip


model, _, processor = open_clip.create_model_and_transforms('hf-hub:woweenie/open-clip-vit-h-nsfw-finetune')
model.eval()


dummy_input = {
    "input_ids": torch.randint(0, 100, (1, 77)),  # Variable length input
}

dynamic_axes = {
    "input_ids": {0: "batch_size", 1: "seq_length"},
}

torch.onnx.export(
    model,
    {"text": dummy_input["input_ids"]},
    "open_clip_text.onnx",
    input_names=["input_ids"],
    output_names=["output"],
    dynamic_axes=dynamic_axes
)

