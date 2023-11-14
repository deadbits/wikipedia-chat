import os

from uuid import uuid4
from loguru import logger
from typing import List, Tuple, Union, Dict
from chromadb import PersistentClient, Settings
from chromadb.utils import embedding_functions


class VectorDB:
    def __init__(self,
        collection_name: str = 'wikichat',
        db_dir: str = 'data/',
        n_results: int = 10,
        openai_api_key: str = None,
        cohere_api_key: str = None
    ) -> None:
        if openai_api_key is not None:
            self.embed_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_key,
                model_name='text-embedding-ada-002'
            )
        elif cohere_api_key is not None:
            self.embed_fn = embedding_functions.CohereEmbeddingFunction(
                api_key=cohere_api_key, 
            model_name='multilingual-22-12'
            )
        else:
            raise Exception('Must provide either an OpenAI or Cohere API key')

        self.db_dir = self.check_directory(db_dir)
        self.collection_name = collection_name
        self.n_results = n_results

        self.client: PersistentClient = PersistentClient(
            path=self.db_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            ),
        )
        self.collection = self.get_or_create_collection(self.collection_name)
        logger.success(f'Loaded database: collection={self.collection_name}')

    def check_directory(self, db_dir: str):
        if db_dir is None:
            raise Exception('Must provide database directory')
        else:
            if not os.path.exists(db_dir):
                logger.info(f'Creating directory: {db_dir}')
                try:
                    os.makedirs(db_dir)
                except Exception as err:
                    logger.error(f'Failed to create database directory: {err}')
                    raise err

            return db_dir

    def get_or_create_collection(self, name: str):
        self.collection = self.client.get_or_create_collection(
            name=name,
            embedding_function=self.embed_fn,
            metadata={'hnsw:space': 'cosine'}
        )
        return self.collection

    def count(self) -> int:
        logger.info('Returning document count')
        return self.collection.count()

    def get(self) -> Dict[str, List[Union[str, List[float], dict]]]:
        logger.info('Getting all documents')
        return self.collection.get()

    def add_texts(self, texts: List[str], metadatas: List[dict]) -> Tuple[bool, List[str]]:
        success = False
        logger.info(f'Adding {len(texts)} texts')

        chunk_ids = {str(i): str(uuid4()) for i in range(len(texts))}

        try:
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=list(chunk_ids.values())
            )
            success = True
        except Exception as err:
            logger.error(f'Failed to add texts to collection: {err}')
            raise err

        return (success, chunk_ids)

    def add_embeddings(self, texts: List[str], embeddings: List[List], metadatas: List[dict]):
        success = False
        logger.info(f'Adding {len(texts)} embeddings')

        chunk_ids = {str(i): str(uuid4()) for i in range(len(texts))}

        try:
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=list(chunk_ids.values())
            )
            success = True
        except Exception as err:
            logger.error(f'Failed to add embeddings to collection: {err}')
            raise err

        return (success, chunk_ids)

    def query(self, text: str) -> List[dict]:
        documents = []
        logger.info(f'Querying database for: {text}')

        try:
            results = self.collection.query(
                query_texts=[text],
                n_results=self.n_results)

            logger.info(f'Found {len(results["ids"][0])} results')

            for node_id, text, metadata, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                documents.append({
                    'id': node_id,
                    'text': text,
                    'metadata': metadata,
                    'distance': distance
                })

            return documents

        except Exception as err:
            logger.error(f'Failed to query database: {err}')
            raise err

