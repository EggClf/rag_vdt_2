from qdrant_client import QdrantClient, models

from fastembed import SparseTextEmbedding, TextEmbedding
from copy import deepcopy
from uuid import uuid4



import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from dotenv import load_dotenv
load_dotenv(os.path.join(current_dir, '..', '.env'))

from llm import OpenAIWrapper,ChatGPT




QDRANT_HOST = "http://localhost:6333"
if os.getenv('QDRANT_HOST') is not None:
    QDRANT_HOST = os.getenv('QDRANT_HOST')

COLLECTION_NAME = "demo_multihop"



def format_docs(docs):
    return "\n--------------\n".join([doc.payload['text'] for doc in docs])


system_prompt = (
    "Below is a question followed by some context from different sources. "
    "Please answer the main question based on the context in <question> tag. The answer to the question is a word or entity. "
    "If the provided information is sufficient to answer the question, respond 'Yes' and answer the question "
    "Else, return no and provide new sub-question needed to answer the original question. The sub question should be simple Does not make sub-question similar the previous question\n\n"
    "Think step-by-step and return in the following format \n\n"
    "## Reasoning:\n"
    "Step-by-step reasoning\n"
    "## Decision:\n"
    "Yes/No\n"
    "## Details:\n"
    "Final Answer/New question.\n\n"
    "Note:\n"
    "- Do not guess early. Use previous steps and retrieved knowledge to build your reasoning gradually."
)


class Chat:
    def __init__(self, 
                 collection_name: str = COLLECTION_NAME, 
                 qdrant_host: str = QDRANT_HOST,
                 device: str = 'cuda',
                 model_name: str = "qwen/qwen2.5-7b-instruct"
                 ):
        
        self.model_name = model_name
        self.client = QdrantClient(qdrant_host, api_key=os.getenv('QDRANT_API_KEY'))
        self.collection_name = collection_name
        self.device = device
        
        if device == 'cuda':
            from sentence_transformers import SentenceTransformer 
            self.dense_embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5").to(device)
        else:
            self.dense_embedding_model = TextEmbedding("BAAI/bge-base-en-v1.5")
        self.bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")
        self.llm = self.init_llm(model_name)
        
      
    def init_llm(self, model_name):
        self.llm = OpenAIWrapper(
            model_name=model_name,
            host = "https://integrate.api.nvidia.com/v1",
            api_key=os.getenv('NVIDIA_API_KEY'),
        )
        
        # self.llm = ChatGPT(
        #     model_name=model_name,
        # )
        
        return self.llm
      
        
    def embedding_func(self, documents):
        if not isinstance(documents, list):
            documents = [documents]

        if self.device == 'cuda':
            dense_embedding = list(self.dense_embedding_model.encode(documents))
        
        else:
            dense_embedding = list(self.dense_embedding_model.embed(documents))
        sparse_embedding = list(self.bm25_embedding_model.query_embed(documents))

        # dense_embedding = [list(embd) for embd in dense_embedding]

        return {
            'dense_embedding': dense_embedding,
            'sparse_embedding': sparse_embedding
        }
        
        
    def rag(self, query, query_filter = None):
        query_embedding = self.embedding_func(query)

        prefetch = [
            models.Prefetch(
                query=query_embedding['dense_embedding'][0],
                using="dense",
                limit=20,
            ),
            models.Prefetch(
                # query=models.SparseVector(**query_embedding['sparse_embedding'][0].as_object()),
                query = query_embedding['sparse_embedding'][0].as_object(),
                using="sparse",
                limit=20,
            ),
        ]
        
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF  # we are using reciprocal rank fusion here
            ),
            prefetch=prefetch,
            query_filter=query_filter,
            limit=5
        ).points

        return format_docs(search_results)
    
    
    
    @staticmethod
    def get_decision(response):

        decision = False
        detail = ''
        
        if '## Decision:' in response and '## Details:' in response:
            decision_text = response.split('## Decision:')[1].split('## Details:' )[0]
            detail = response.split('## Details:' )[1]

            if 'yes' in decision_text.lower():
                decision = True
            else:
                decision = False

        return decision, detail


    def force_answer(self, query, rag_content, messages = []):

        messages.append(
            {
                'role':'user',
                'content': f'<question>{query}</question> <content>{rag_content}<context>. Based on the given context, think step-by-step and return your answer.'
            }
    )

        return self.llm.stream(messages)

    def solve(self, query, previous_messages = [], max_iter = 3):

        
        messages = [
            {
                'role' : 'system',
                'content': system_prompt
            },
        ]
        
        conversation = deepcopy(messages)
        
        if len(conversation) > 0:
            if conversation[0]['role'] == 'system':
                conversation.pop(0)
                
            messages.extend(deepcopy(conversation))
        
        messages.append(
            {
                'role' : 'user',
                'content': 'decoy'
            }
        )

        decision = False
        intermediate_question = query

        rag_content = ''
        response = 'Anw cut'
        
        yield '\n<think>\n'
        
        for i in range(max_iter):
            # remove prev
            rag_content += self.rag(intermediate_question)
            
            content = f"Here is the provided content: \n\n {rag_content}. Answer the followning question \n\n<question>\n\n{query}\n\n</question>"

            messages.append(
                {
                    'role' : 'user',
                    'content': content
                }
            )

            response = ''
            stream_response = self.llm.stream(messages)
            for chunk in stream_response:
                if isinstance(chunk, str):
                    response += chunk
                    yield chunk
            
            # response = self.llm(messages)
            
            # print('### =================', i + 1, '================ ###')
            # print(response)
            # print('### ===================================== ###')

            decision, detail = self.get_decision(response)

            messages.append(
                {
                    'role':'assistant',
                    'content': response
                }
            )
            

            intermediate_question = detail
            if decision:
                break
          
        yield '\n</think>\n'  


        for chunk in self.force_answer(query, rag_content, messages=previous_messages):
            if isinstance(chunk, str):
                yield chunk
    
    def stream(self, messages, model_name = None):
        if model_name is not None:
            self.init_llm(model_name)
        
        query = messages[-1]['content']
        messages = messages[:-1]
        
        return self.solve(query, messages)

        
if __name__ == "__main__":
    chat = Chat(model_name='gpt-4.1-nano')
    
    query = "Who is the CEO of NVIDIA?"
    
    messages = [

        {
            'role': 'user',
            'content': query
        }
    ]
    
    for response in chat.stream(messages):
        print(response, end='', flush=True)