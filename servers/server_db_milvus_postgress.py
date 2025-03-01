from typing import Optional, Annotated

from fastapi import FastAPI, Response, Cookie, Request
from pydantic import BaseModel, Field, ConfigDict
from pymilvus import Collection, connections
import random
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import uuid
import threading
from psycopg_pool import AsyncConnectionPool
from contextlib import asynccontextmanager
import os
#TODO: change to grequests
import requests

#TODO: make config with this values
#TODO: add docker-compose
SERVER_DB_PORT = int(os.getenv("SERVER_DB_PORT"))
POSTGRESS_PORT = int(os.getenv("POSTGRESS_PORT"))
DB_MILVUS_PORT = int(os.getenv("DB_MILVUS_PORT"))
CURRENT_HOST_PORT = int(os.getenv("CURRENT_HOST_PORT"))


POSTGRESS_DB_NAME = os.getenv("POSTGRESS_DB_NAME")
POSTGRESS_USER_NAME = os.getenv("POSTGRESS_USER_NAME")
POSTGRESS_USER_PASSWORD = os.getenv("POSTGRESS_USER_PASSWORD")

TEMP_CURRENT_HOST = os.getenv("LOCAL_HOST_ADDRESS", "localhost")
TEMP_HOST_NAME_POSTGRESS = os.getenv("TEMP_HOST_NAME_POSTGRESS", "fd301d06fec2")
TEMP_HOST_NAME_MILVUS = os.getenv("TEMP_HOST_NAME_MILVUS", "ac5a7858b1bd")
TEMP_HOST_NAME_EMBEDDER = os.getenv("TEMP_HOST_NAME_MILVUS", TEMP_CURRENT_HOST)

SERVER_EMBEDDER_PORT = int(os.getenv("SERVER_EMBEDDER_PORT"))
SERVER_ADDRESS_EMBEDDER = TEMP_HOST_NAME_EMBEDDER
SERVER_IMAGE_EMBEDDING = f"{SERVER_ADDRESS_EMBEDDER}:{SERVER_EMBEDDER_PORT}/get_image_embedding"
SERVER_TEXT_EMBEDDING = f"{SERVER_ADDRESS_EMBEDDER}:{SERVER_EMBEDDER_PORT}/get_text_embedding"


DB_MILVUS_SERVER = f"http://{TEMP_HOST_NAME_MILVUS}:{DB_MILVUS_PORT}"



IMG_CONTENT_BY_IMAGE_EMBEDDING = "img_content_by_image_embedding"
IMG_CONTENT_BY_IMAGE_EMBEDDING_LIGHT = "img_content_by_image_embedding_ml_mobile_clip_s0"
VIDEO_CONTENT_BY_IMAGE_EMBEDDING = "video_content_by_image_embedding"
LIMIT_SEARCH = 100
NPROBE = 30

CAPTION_LIMIT_SIZE = 200

CONTENT_ID = "content_id"
EMBEDDING = "embedding"


USE_LAST_N_LIKES = 20


CAPTION = "text_caption"
URL = "url"
VIDEO_URL = "video_url"
POST_URL = "post_url"

INDEX_FROM_MILVUS = "index_from_milvus"
LIKE = "like"
CHAT_ID = "chat_id"

LIMIT_INDS_TO_RANDOM_RECOMMENDER = 200
LIMIT_SEARCH_FOR_ONE_EMBEDDING = 15
LIMIT_SEARCH_FOR_COOKIE_RECOMMENDER = 100


# SOME CONNECTION INIT
def get_conn_str():
    return f"""
    dbname={POSTGRESS_DB_NAME}
    user={POSTGRESS_USER_NAME}
    password={POSTGRESS_USER_PASSWORD}
    host={TEMP_HOST_NAME_POSTGRESS}
    port={POSTGRESS_PORT}
    """

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.async_pool = AsyncConnectionPool(conninfo=get_conn_str())
    yield
    await app.async_pool.close()


app = FastAPI()

app = FastAPI(debug=False,
              lifespan=lifespan)


origins = [
    f"{TEMP_CURRENT_HOST}:{CURRENT_HOST_PORT}",
    f"{TEMP_CURRENT_HOST}",
    f'{TEMP_CURRENT_HOST}:{CURRENT_HOST_PORT}/get_random_cookies',
    f"{TEMP_CURRENT_HOST}:{CURRENT_HOST_PORT}/random_images_for_cookies",
    f"{TEMP_CURRENT_HOST}:{CURRENT_HOST_PORT}/recommend_by_text_for_cookies",
    f"{TEMP_CURRENT_HOST}:{CURRENT_HOST_PORT}/recommend_by_id_for_cookies",
    f"{TEMP_CURRENT_HOST}:{CURRENT_HOST_PORT}/get_liked_for_cookies",
    "null"
]



# FOR ROUTING ALL REQUEST

class Embedding(BaseModel):
    embedding: list[float]


class ImageQueryRaw(BaseModel):
    image: str
    used_ids: list[int] = []
    limit: int = 30

class TextQueryRaw(BaseModel):
    text: str
    used_ids: list[int] = []
    limit: int = 30


@app.get("/get_image_embedding")
async def get_image_embedding(e: ImageQueryRaw) -> Embedding:
    return requests.get(SERVER_IMAGE_EMBEDDING, json={"image": e.image}).json()

@app.get("/get_text_embedding")
async def get_text_embedding(e: TextQueryRaw) -> Embedding:
    return requests.get(SERVER_TEXT_EMBEDDING, json={"text": e.text}).json()


class ResultData(BaseModel):
    image_url: str
    text_caption: str
    id: str
    video_url: str
    post_url: str


class ListOfResultData(BaseModel):
    list_of_data: list[ResultData]


async def full_text_search(request, text_string, ts_index="ts", columns="content_id,extra_tags_description,urls_main_url,urls_affilated_url",
                            table_name="img_content"):
    async with request.app.async_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(f"SELECT  {columns} FROM {table_name} WHERE {ts_index} @@ phraseto_tsquery('english', '{text_string}') LIMIT 100;")
            result_list = [{col_name: v for v, col_name in zip(row, columns.split(","))} for row in await cur.fetchall()]
            returned_data = []
            for r in result_list:
                returned_data.append({IMG_URL: r["urls_main_url"],\
                TEXT_CAPTION: r["extra_tags_description"][:CAPTION_LIMIT_SIZE], \
                ID: str(r["content_id"]), \
                POST_URL: r["urls_affilated_url"],
                VIDEO_URL: ""})

            return {"list_of_data":  returned_data}


@app.get("/find_data_by_image")
async def find_data_by_image(query: ImageQueryRaw, request: Request) -> ListOfResultData:
    embedding = (await get_image_embedding(query))["embedding"]
    return await get_img_content_from_db([embedding], limit=query.limit, request=request, filter_query_ids=query.used_ids)


@app.get("/find_data_by_text")
async def find_data_by_text(query: TextQueryRaw, request: Request) -> ListOfResultData:
    embedding = (await get_text_embedding(query))["embedding"]
    by_text_search = await full_text_search(request, query.text)
    by_text_search_title = await full_text_search(request, query.text, ts_index="ts_extra_tags_title_img")

    by_embedding_search = await get_img_content_from_db([embedding], limit=query.limit, request=request, filter_query_ids=query.used_ids)
    full_list = by_text_search["list_of_data"] + by_embedding_search["list_of_data"] + by_text_search_title["list_of_data"]
    random.shuffle(full_list)
    return {"list_of_data": full_list}


@app.get("/find_video_by_image")
async def find_video_by_image(query: ImageQueryRaw, request: Request) -> ListOfResultData:
    embedding = (await get_image_embedding(query))["embedding"]
    return await get_video_content_from_db([embedding], limit=query.limit, request=request,
    filter_query_ids=query.used_ids)


@app.get("/find_video_by_text")
async def find_video_by_text(query: TextQueryRaw, request: Request) -> ListOfResultData:

    embedding = (await get_text_embedding(query))["embedding"]
    by_text_search = await full_text_search(request, query.text, table_name="video_content", ts_index="ts_video")

    by_embedding_search =await get_video_content_from_db([embedding], limit=query.limit, request=request, filter_query_ids=query.used_ids)
    full_list = by_text_search["list_of_data"] + by_embedding_search["list_of_data"]
    random.shuffle(full_list)

    return {"list_of_data": full_list} 


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

@app.get("/")
async def main():
    return {"message": "Hello World"}


class ImageQuery(BaseModel):
    embedding: list[float]
    limit: int = LIMIT_SEARCH
    used_ids: Optional[list[str]] = []


IMG_URL = "image_url"
TEXT_CAPTION = "text_caption"
ID = "id"
POST_URL = "post_url"
VIDEO_URL = "video_url"



async def select_columns_from_psql(columns="extra_tags_description,urls_main_url,urls_affilated_url",
                                   table_name="img_content",
                                   content_ids=None,
                                   request=None):
    content_ids = ",".join([f"'{a}'" for a in list(set(content_ids))])
    async with request.app.async_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(f"SELECT  {columns} FROM {table_name} WHERE content_id IN ({content_ids})")
            result_list = [{col_name: v for v, col_name in zip(row, columns.split(","))} for row in await cur.fetchall()]
            return result_list


async def get_of_content_ids(ids, request):
    alias = f"current_thread_{threading.current_thread().ident}"                 
    connections.connect(uri=DB_MILVUS_SERVER, alias=alias)
    results = await select_columns_from_psql(columns="extra_tags_description,urls_main_url,urls_affilated_url,content_id",
                                   table_name="video_content",
                                   content_ids=ids,
                                   request=request)
    returned_data = []
    for r in results:
        returned_data.append({IMG_URL: r["urls_main_url"],\
        TEXT_CAPTION: r["extra_tags_description"][:CAPTION_LIMIT_SIZE], \
        ID: str(r["content_id"]), \
        POST_URL: r["urls_affilated_url"],
        VIDEO_URL: ""})
    
    results = await select_columns_from_psql(columns="extra_tags_description,urls_main_url,urls_affilated_url,content_id",
                                   table_name="img_content",
                                   content_ids=ids,
                                   request=request)
    for r in results:
        returned_data.append({IMG_URL: r["urls_main_url"],\
        TEXT_CAPTION: r["extra_tags_description"][:CAPTION_LIMIT_SIZE], \
        ID: str(r["content_id"]), \
        POST_URL: r["urls_affilated_url"],
        VIDEO_URL: ""})
    return {"list_of_data": returned_data}


async def get_embeddings_of_content_ids(content_ids, name_of_server=IMG_CONTENT_BY_IMAGE_EMBEDDING):
    alias = f"current_thread_{threading.current_thread().ident}"                 
    connections.connect(uri=DB_MILVUS_SERVER, alias=alias)
    collection = Collection(name=name_of_server, using=alias)
    content_ids = [int(a) for a in content_ids]
    results = collection.query(
        expr=f"{CONTENT_ID} in {content_ids}",
        output_fields=[EMBEDDING]
    )
    embeddings = [r[EMBEDDING] for r in results]
    return embeddings

#TODO: fi this nonsense
TRY_TO_SEARCH = 5
async def get_content_from_db(embeddings, limit=LIMIT_SEARCH, 
                              filter_query_ids=None, 
                              table_name=IMG_CONTENT_BY_IMAGE_EMBEDDING, psql_table_name="img_content", 
                              request=None,
                              **kwargs):
    alias = f"current_thread_{threading.current_thread().ident}"                 
    connections.connect(uri=DB_MILVUS_SERVER, alias=alias)
    collection = Collection(name=table_name, using=alias)
    print(filter_query_ids, "in get_content_from_db")
    if filter_query_ids is None:
        filter_query_ids = [-1]
    for k in range(1, TRY_TO_SEARCH + 1):
        results = collection.search(
            anns_field=EMBEDDING,
            data=embeddings,
            output_fields=[CONTENT_ID],
            filter=f"!(content_id IN ({','.join([str(a) for a in list(filter_query_ids)])}))",
            param={"metric_type": "COSINE", "params": {"nprobe": NPROBE},},
            limit=min(LIMIT_SEARCH * k, limit * k),
            **kwargs
        )
        filter_query_ids = set([int(a) for a in filter_query_ids])
        ids = []
        for hits in results:
            for hit in hits:
                if int(hit.entity.get(CONTENT_ID)) not in filter_query_ids:
                    ids.append(hit.entity.get(CONTENT_ID))
        
        ids = list(set(ids))
        if ids:
            break
    results = await select_columns_from_psql(columns="extra_tags_description,urls_main_url,urls_affilated_url,content_id",
                                   table_name=psql_table_name,
                                   content_ids=ids,
                                   request=request)
    returned_data = []
    for r in results:
        returned_data.append({IMG_URL: r["urls_main_url"],\
        TEXT_CAPTION: r["extra_tags_description"][:CAPTION_LIMIT_SIZE], \
        ID: str(r["content_id"]), \
        POST_URL: r["urls_affilated_url"],
        VIDEO_URL: ""})
    print(returned_data)
    return {"list_of_data": returned_data}



async def get_img_content_from_db(embeddings, limit=LIMIT_SEARCH, filter_query_ids=None, request=None, table_name=IMG_CONTENT_BY_IMAGE_EMBEDDING,
                                  **kwargs):
    return await get_content_from_db(embeddings=embeddings, limit=limit, filter_query_ids=filter_query_ids,
                                     psql_table_name="img_content", request=request,
                                     table_name=table_name)

async def get_video_content_from_db(embeddings, limit=LIMIT_SEARCH, filter_query_ids=None, request=None, table_name=VIDEO_CONTENT_BY_IMAGE_EMBEDDING, **kwargs):
    return await get_content_from_db(embeddings=embeddings, limit=limit, filter_query_ids=filter_query_ids,
                                     table_name=table_name, psql_table_name="video_content",
                                     request=request)

@app.get("/find_data_by_image_embedding")
async def find_data_by_image_embedding(query: ImageQuery, request: Request) -> ListOfResultData:
    return await get_img_content_from_db([query.embedding], limit=query.limit, request=request,
    filter_query_ids=query.used_ids)


@app.get("/find_data_by_text_embedding")
async def find_data_by_text_embedding(query: ImageQuery, request: Request) -> ListOfResultData:
    return await get_img_content_from_db([query.embedding], limit=query.limit, request=request,
    filter_query_ids=query.used_ids)


@app.get("/find_video_by_image_embedding")
async def find_data_by_image_embedding(query: ImageQuery, request: Request) -> ListOfResultData:
    return await get_video_content_from_db([query.embedding], limit=query.limit, request=request,
    filter_query_ids=query.used_ids)


@app.get("/find_video_by_text_embedding")
async def find_data_by_text_embedding(query: ImageQuery, request: Request) -> ListOfResultData:
    return await get_video_content_from_db([query.embedding], limit=query.limit, request=request,
    filter_query_ids=query.used_ids)
    

class LikeData(BaseModel):
    index_from_milvus: int
    like: int
    chat_id: int


async def insert_likes_from_psql(content_id=-1,
                                   is_like=1,
                                   chat_id=-1,
                                   request=None):

    async with request.app.async_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(f'INSERT INTO "likes" VALUES {(str(content_id), is_like, str(chat_id))};')


@app.post("/add_like")
async def add_like(like_data: LikeData, request: Request):
    try:
        await insert_likes_from_psql(content_id=like_data.index_from_milvus,
                                   is_like=like_data.like,
                                   chat_id=like_data.chat_id,
                                   request=request)
    except Exception as e:
        print(f"Db failed with exception as {e}")


class LikeDataWeb(BaseModel):
    index_from_milvus: str
    like: int
    chat_id: str

@app.post("/add_like_web")
async def add_like_web(like_data: LikeDataWeb, request: Request):
    try:
        await insert_likes_from_psql(content_id=like_data.index_from_milvus,
                                   is_like=like_data.like,
                                   chat_id=like_data.chat_id,
                                   request=request)
    except Exception as e:
        print(f"Db failed with exception as {e}")


class UserHistory(BaseModel):
    liked_ids: list[int]
    dislike_ids: list[int]
    shown_ids: list[int]


@app.get("/recommend")
async def recommend(meta_info: UserHistory, request: Request) -> ListOfResultData:
    embeddings = await get_embeddings_of_content_ids(meta_info.liked_ids)
    if len(embeddings) < USE_LAST_N_LIKES:
        embeddings = (embeddings + [np.random.rand(EMBEDDING_SIZE) for _ in range(USE_LAST_N_LIKES)])[-USE_LAST_N_LIKES:]
    return await get_img_content_from_db(embeddings, limit=LIMIT_SEARCH, request=request,
                                         filter_query_ids=meta_info.shown_ids)


RANDOM_IMAGES_COUNT = 100
EMBEDDING_SIZE = 1024
@app.get("/random_images")
async def random_images(request: Request) -> ListOfResultData:
    embeddings = [np.random.rand(EMBEDDING_SIZE) for _ in range(RANDOM_IMAGES_COUNT)]
    return await get_img_content_from_db(embeddings, limit=LIMIT_SEARCH, request=request)
    

@app.get("/get_random_cookies")
async def get_random_cookies(response: Response):
    value = uuid.uuid4()
    response.set_cookie(key='user_id', value=str(value), httponly=False, secure=True, 
        samesite=None) 
    print(value, "get_random_cookies")
    return {'user_id': str(value)}



class UserId(BaseModel):
    user_id: str

TIMEOUT_FOR_RECOMMENDATIONS_PER_ONE = 10

#TODO rename chat_it -> chat_id
async def get_likes(chat_id, is_like=1, request=None):
    async with request.app.async_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(f"SELECT content_id FROM likes WHERE chat_it = '{chat_id}' AND is_like = {is_like} ;")
            result_list = list(set([row[0] for row in await cur.fetchall()])) 
            return result_list


@app.post("/random_images_for_cookies")
async def random_images_for_cookies(user: UserId, request: Request) -> ListOfResultData:
    user_id = user.user_id
    print(user_id)
    liked_ids = []
    all_data = []
    if user_id is not None:
        likes = await get_likes(user_id, is_like=1, request=request)
        liked_ids = likes
        all_data = liked_ids[:]
        # TODO: work separately with likes and dislikes
        # for r in likes:
        #     if r[LIKE] > 0:
        #         liked_ids.append(r["index_from_milvus"])
        #     if "index_from_milvus" in r:
        #         all_data.append(r["index_from_milvus"])
    embeddings = await get_embeddings_of_content_ids(liked_ids[-USE_LAST_N_LIKES:])
    if len(embeddings) < USE_LAST_N_LIKES:
        embeddings = (embeddings + [np.random.rand(EMBEDDING_SIZE) for _ in range(USE_LAST_N_LIKES)])[-USE_LAST_N_LIKES:]
    return await get_img_content_from_db(embeddings, limit=LIMIT_SEARCH, filter_query_ids=all_data, request=request)



async def get_likes_from_user_id(user_id, request):
    liked_ids = []
    all_data = []
    if user_id is not None:
        likes = await get_likes(user_id, is_like=1, request=request)
        liked_ids = likes
        all_data = liked_ids[:]
        # TODO: work separately with likes and dislikes
        # for r in likes:
        #     if r[LIKE] > 0:
        #         liked_ids.append(r["index_from_milvus"])
        #     if "index_from_milvus" in r:
        #         all_data.append(r["index_from_milvus"])
    return liked_ids, set(all_data)

@app.post("/get_liked_for_cookies")
async def get_liked_for_cookies(user: UserId, request: Request) -> ListOfResultData:
    user_id = user.user_id
    liked_ids, all_data = await get_likes_from_user_id(user_id, request)
    return await get_of_content_ids(liked_ids, request)



class ImageId(BaseModel):
    image_id: str


class OneImageRecommenderQuery(BaseModel):
    user: UserId
    image: ImageId

@app.post("/recommend_by_id_for_cookies")
async def recommend_by_id_for_cookies(query: OneImageRecommenderQuery, request: Request) -> ListOfResultData:
    user_id = query.user.user_id
    liked_ids, all_data = await get_likes_from_user_id(user_id, request)

    embeddings = await get_embeddings_of_content_ids([int(query.image.image_id)])
    if len(embeddings) < USE_LAST_N_LIKES:
        embeddings = (embeddings + [np.random.rand(EMBEDDING_SIZE) for _ in range(USE_LAST_N_LIKES)])[-USE_LAST_N_LIKES:]
    
    return await get_img_content_from_db(embeddings, limit=LIMIT_SEARCH, filter_query_ids=all_data, request=request)



class OneEmbeddingRecommenderQuery(BaseModel):
    user: UserId
    embedding: ImageQuery

@app.post("/recommend_by_text_for_cookies")
async def recommend_by_text_for_cookies(query: OneEmbeddingRecommenderQuery, request: Request) -> ListOfResultData:
    user_id = query.user.user_id
    liked_ids, all_data = await get_likes_from_user_id(user_id)
    return await get_img_content_from_db([query.embedding.embedding], limit=LIMIT_SEARCH, filter_query_ids=all_data,
                                         request=request)