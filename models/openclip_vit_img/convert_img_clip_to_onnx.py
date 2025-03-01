from PIL import Image
import torch
import io
import base64
from io import BytesIO

import open_clip


model, _, processor = open_clip.create_model_and_transforms('hf-hub:woweenie/open-clip-vit-h-nsfw-finetune')
model.eval()
tokenizer = open_clip.get_tokenizer('hf-hub:woweenie/open-clip-vit-h-nsfw-finetune')
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "open_clip.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

