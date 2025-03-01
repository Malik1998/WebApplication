
FROM python:3.9

WORKDIR /code

COPY servers/requirements.txt /code/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


COPY servers/server_db_milvus_postgress.py /code/server_db_milvus_postgress.py
COPY servers/__init__.py /code/__init__.py


CMD ["fastapi", "run", "/code/server_db_milvus_postgress.py", "--port", "8180"]