from fastapi import FastAPI
from pydantic import BaseModel

from PIL import Image
import torch
import io
import base64
from io import BytesIO

import open_clip
from anyio.lowlevel import RunVar
from anyio import CapacityLimiter
from fastapi.middleware.cors import CORSMiddleware
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput


triton_client = InferenceServerClient(url="localhost:8000")
model, _, processor = open_clip.create_model_and_transforms('hf-hub:woweenie/open-clip-vit-h-nsfw-finetune')


similiarity = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
BATCH_SIZE = 16

SERVER_EMBEDDER_PORT = 8111



def get_image_embedding_v2(image):
    inputs = processor(image).unsqueeze(0).numpy()
    input_tensor = InferInput("input", inputs.shape, "FP32")
    input_tensor.set_data_from_numpy(inputs)

    output_tensor = InferRequestedOutput("output")

    # Выполнение инференса
    response = triton_client.infer(
        model_name="open_clip_image",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )
    output = response.as_numpy("output")[0]
    output = (output) / (output ** 2).sum()
    output = list(output)
    return output

# def get_image_embedding_v2(image):
#     model, _, processor = open_clip.create_model_and_transforms('hf-hub:woweenie/open-clip-vit-h-nsfw-finetune')
#     model.eval()
#     tokenizer = open_clip.get_tokenizer('hf-hub:woweenie/open-clip-vit-h-nsfw-finetune')
#     with torch.no_grad():
#         inputs = processor(image).unsqueeze(0)
#         img_features = model.encode_image(inputs)
#         img_features /= img_features.norm(dim=-1, keepdim=True)
#     return img_features.flatten().tolist()


def get_text_embeddings_v2(text):
    model, _, _ = open_clip.create_model_and_transforms('hf-hub:woweenie/open-clip-vit-h-nsfw-finetune')
    model.eval()
    tokenizer = open_clip.get_tokenizer('hf-hub:woweenie/open-clip-vit-h-nsfw-finetune')
    with torch.no_grad():
        inputs = tokenizer(text)
        text_features = model.encode_text(inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.flatten().tolist()

def get_text_embeddings_v1(text):
    model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained='checkpoints/mobileclip_s0.pt')
    tokenizer = mobileclip.get_tokenizer('mobileclip_s0')
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text)
        text_features = model.encode_text(inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.flatten().tolist()


def calculate_similiarity(text_embedding, image_embedding):
    return similiarity(text_embedding, image_embedding)



app = FastAPI()


origins = [
    "http://localhost:8111",
    "http://localhost",
    'http://localhost:8111/get_text_embedding_2',
    "http://localhost:8111",
    "http://localhost:8111/get_text_embedding_2",
    "http://localhost:8111/recommend_by_text_for_cookies",
    'null'
]   

@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = ", ".join(origins)
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.on_event("startup")
# def startup():
#     print("start")
#     RunVar("_default_thread_limiter").set(CapacityLimiter(4))


class Embedding(BaseModel):
    embedding: list[float]


class ImageQuery(BaseModel):
    image: bytes

@app.get("/get_image_embedding")
async def get_image_embedding(e: ImageQuery) -> Embedding:
    embedding = get_image_embedding_v2(Image.open(BytesIO(base64.b64decode(e.image))))
    return {"embedding": embedding}



class EmbeddingDct(BaseModel):
    embedding: list[float]
    content_id: str


class ImageQueryDct(BaseModel):
    image: bytes
    content_id: str


class EmbeddingList(BaseModel):
    embeddings: list[EmbeddingDct]


class ImageQueryList(BaseModel):
    images: list[ImageQueryDct]


def get_image_embedding_v2_dct(dct):
    keys_list = list(dct.keys())
    lst = []
    for i, k in enumerate(keys_list):
        with torch.no_grad():
            lst.append(processor(dct[k]).unsqueeze(0))
        dct[k] = i
    iter_count = (len(lst) // BATCH_SIZE + (len(lst) % BATCH_SIZE != 0))
    for i in range(iter_count):
        # print(f"batch {i}")
        input_data = torch.cat(lst[i * BATCH_SIZE: (i + 1) * BATCH_SIZE], dim=0).numpy()
        input_tensor = InferInput("input", input_data.shape, "FP32")
        input_tensor.set_data_from_numpy(input_data)

        output_tensor = InferRequestedOutput("output")

        # Выполнение инференса
        response = triton_client.infer(
            model_name="open_clip_image",
            inputs=[input_tensor],
            outputs=[output_tensor],
        )
        output = response.as_numpy("output")
        # output /= output.norm(dim=-1, keepdim=True)
        for k in keys_list[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]:
            vec = output[dct[k] - BATCH_SIZE * i]
            vec = (vec) / (vec ** 2).sum()
            dct[k] = list(vec)
    return dct


@app.get("/get_image_embedding_list")
async def get_image_embedding_list(e: ImageQueryList) -> EmbeddingList:
    dct = dict()
    for r in e.images:
        try:
            dct[r.content_id] = Image.open(BytesIO(base64.b64decode(r.image)))
        except Exception as e:
            print(e)
    embedding_dict = get_image_embedding_v2_dct(dct)
    return {"embeddings": [{"embedding": v, "content_id": k} for k, v in embedding_dict.items()]}

@app.post("/get_image_embedding_list")
async def get_image_embedding_list2(e: ImageQueryList) -> EmbeddingList:
    dct = dict()
    for r in e.images:
        try:
            dct[r.content_id] = Image.open(BytesIO(base64.b64decode(r.image)))
        except Exception as e:
            print(e)
    embedding_dict = get_image_embedding_v2_dct(dct)
    return {"embeddings": [{"embedding": v, "content_id": k} for k, v in embedding_dict.items()]}

class Text(BaseModel):
    text: str

@app.get("/get_text_embedding")
async def search(e: Text) -> Embedding :
    embedding = get_text_embeddings_v2(e.text)
    return {"embedding": embedding}

@app.get("/get_text_embedding_light")
async def search(e: Text) -> Embedding :
    embedding = get_text_embeddings_v1(e.text)
    return {"embedding": embedding}

@app.post("/get_text_embedding_2")
async def search_2(e: Text) -> Embedding :
    embedding = get_text_embeddings_v2(e.text)
    return {"embedding": embedding}
