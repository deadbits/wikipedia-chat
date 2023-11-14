import os
import sys
from tqdm import tqdm
from loguru import logger
from datasets import load_dataset

from wikichat.vectordb import VectorDB
from wikichat.models import Document


cohere_api_key = os.environ['COHERE_API_KEY']

vector_db = VectorDB(
    cohere_api_key=cohere_api_key
)

try:
    docs_stream = load_dataset(
        'Cohere/wikipedia-22-12-simple-embeddings',
        split="train",
        streaming=True
    )
except Exception as err:
    logger.error(f'Error loading dataset: {err}')
    raise

logger.info('Reading documents from dataset ...')

max_docs = 485860
chunk_size = 100
docs_chunk = []

for doc in tqdm(docs_stream, total=max_docs):
    new = Document(
        docid=doc['id'],
        text=doc['text'],
        title=doc['title'],
        emb=doc['emb']
    )
    docs_chunk.append(new)

    if len(docs_chunk) % 100 == 0:
        logger.info(f'Processing document: {new.docid} {new.title}')

    if len(docs_chunk) >= chunk_size:
        logger.info(f'Adding {chunk_size} documents to database.')
        texts = [doc.text for doc in docs_chunk]
        embeddings = [doc.emb for doc in docs_chunk]
        metadatas = [{'docid': doc.docid, 'title': doc.title} for doc in docs_chunk]
        
        vector_db.add_embeddings(texts=texts, embeddings=embeddings, metadatas=metadatas)
        docs_chunk.clear()

if len(docs_chunk) > 0:
    logger.info(f'Adding {len(docs_chunk)} documents to database.')
    texts = [doc.text for doc in docs_chunk]
    embeddings = [doc.emb for doc in docs_chunk]
    metadatas = [{'docid': doc.docid, 'title': doc.title} for doc in docs_chunk]

    vector_db.add_embeddings(texts=texts, embeddings=embeddings, metadatas=metadatas)

logger.info('Finished adding documents to database.')
logger.info(f'Document count: {vector_db.count()}')
