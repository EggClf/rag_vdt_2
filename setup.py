from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from uuid import uuid4
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv()
import os


QDRANT_HOST = "http://localhost:6333"
COLLECTION_NAME = "demo_multihop"

def load_corpus(qdrant_host=QDRANT_HOST, collection_name=COLLECTION_NAME, api_key=None):
    corpus_data = load_dataset("yixuantt/MultiHopRAG", "corpus")["train"]
    





    qdrant = QdrantClient(qdrant_host, api_key= api_key)

    # Create collection
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config= {
            'dense': models.VectorParams(size=768, distance=models.Distance.COSINE)
        },
        sparse_vectors_config= { 'sparse': models.SparseVectorParams(modifier=models.Modifier.IDF)}

    )

    dense_embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5").to('cuda')
    bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")

    def embedding_func(documents):
        if not isinstance(documents, list):
            documents = [documents]

        dense_embedding = list(dense_embedding_model.encode(documents))
        # dense_embedding = list(dense_embedding_model.embed(documents))
        sparse_embedding = list(bm25_embedding_model.embed(documents))

        dense_embedding = [list(embd) for embd in dense_embedding]

        return {
            'dense_embedding' : dense_embedding,
            'sparse_embedding' : sparse_embedding
        }
        
        

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    # Convert to LangChain Document format and split each doc into chunks

    documents = []
    # i = 0

    BS = 128

    for doc in corpus_data:
        # i += 1
        # if i > 100:
        #     break
        payload = {
            "title": doc.get("title", ""),
            "source": doc.get("source", ""),
            "published_at": doc.get("published_at", ""),
        }
        # Split the text into chunks
        splits = text_splitter.split_text(doc["body"])
        # Create a Document object for each chunk
        for chunk in splits:
            c_payload = payload.copy()
            c_payload['text'] = chunk

            documents.append(c_payload)


    for i in range(0, len(documents), BS):

        points = []
        batch = documents[i:i+BS]
        embeddings = embedding_func([payload['text'] for payload in batch])
        # print(embeddings)

        for payload, sparse_embedding, dense_embedding in zip(batch, embeddings['sparse_embedding'], embeddings['dense_embedding']):
            points.append(
                models.PointStruct(id=str(uuid4()), 
                                    vector={
                                        'sparse' : models.SparseVector(**sparse_embedding.as_object()),
                                        # 'sparse' : sparse_embedding.as_object(),
                                        'dense' : dense_embedding
                                    }, 
                                    payload=payload)
            )

        qdrant.upsert(collection_name=collection_name, points=points)
        
        
if __name__ == "__main__":
    host = os.getenv("QDRANT_HOST", QDRANT_HOST)
    api_key = os.getenv("QDRANT_API_KEY", None)
    load_corpus(qdrant_host=host, collection_name=COLLECTION_NAME, api_key=api_key)
    print(f"Corpus loaded into Qdrant collection '{COLLECTION_NAME}' at {host}")