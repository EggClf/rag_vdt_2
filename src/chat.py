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
from src.ircot_agent import ircot_pipeline_agent



QDRANT_HOST = "http://localhost:6333"
if os.getenv('QDRANT_HOST') is not None:
    QDRANT_HOST = os.getenv('QDRANT_HOST')

COLLECTION_NAME = "demo_multihop"


def _format_doc(doc):
    
    return f"Title: {doc.get("title", "")}\n\nSource: {doc.get("source", "")}\n\nPublished: {doc.get("published_at", "")}\n\nContent: {doc.get('text', '')}"

def format_docs(docs):
    return "\n--------------\n".join([_format_doc(doc.payload) for doc in docs])


def prune_think_tag(messages):
    """
    Remove the <think> tag and its content from the text.
    """
    for msg in messages:
        if 'content' in msg and '<think>' in msg['content']:
            msg['content'] = msg['content'].split('</think>')[1]
            
    return messages

system_prompt = (
    "Below is a question followed by some context from different sources. "
    "Please answer the main question based on the context in <question> tag. The answer to the question is a word or entity. "
    "If the provided information is sufficient to answer the question, respond 'Yes' and answer the question "
    "Else, return no and provide new sub-question as deconstruction of the question. These new questions will then be used to search for more information, so make it simple and as informative as possible.\n DO NOT make sub-question similar the previous questions\n\n"
    "Think step-by-step and return in the following format \n\n"
    "## Reasoning:\n"
    "Step-by-step reasoning\n"
    "## Decision:\n"
    "Yes/No\n"
    "## Details:\n"
    "- Sub-question 1.\n"
    "- Sub-question 2.\n\n"
    "...\n\n"
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
        
        
    def rag(self, query, query_filter = None, ignore_ids = set()):
        query_embedding = self.embedding_func(query)

        results = []

        for i in range(len(query_embedding['dense_embedding'])):

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
                # query_filter=query_filter,
                limit=5
            ).points
        
            search_results = [doc for doc in search_results if doc.id not in ignore_ids]
            
            for doc in search_results:
                ignore_ids.add(doc.id)
                results.append(doc)

        return format_docs(results), ignore_ids
    
    
    
    @staticmethod
    def get_decision(response):

        decision = False
        detail = []
        
        if '## Decision:' in response and '## Details:' in response:
            decision_text = response.split('## Decision:')[1].split('## Details:' )[0]
            detail = response.split('## Details:' )[1]
            
            # Split the bulleted list
            detail = detail.strip().split('\n')
            detail = [d for d in detail if d.startswith('-')]
            detail = [d.strip() for d in detail if d.strip() != '']

            if 'yes' in decision_text.lower():
                decision = True
            else:
                decision = False

            print(f'\n\n## Decision: {decision_text.strip()}\n')

        return decision, detail


    def final_answer(self, llm, query, rag_content, messages = []):
        
        messages.append(
            {
                'role':'user',
                'content': f'<question>\n{query}\n</question>\n\n <context>\n{rag_content}\n</context>\n\nBased on the given context, think step-by-step and return your answer. the content in <context> tag is hidden from the user. Do not mention it in your answer.'
            }
    )

        for chunk in llm.stream(messages):
            if isinstance(chunk, str):
                yield chunk
    
    
    def basic_solve(self, query, previous_messages = [], max_iter = 3):
        
        
        if len(previous_messages) > 0:
            if previous_messages[0]['role'] == 'system':
                previous_messages.pop(0)
                
            messages.extend(deepcopy(previous_messages))
        
        rag_content, ignore_ids = self.rag(query)
        
        
        for chunk in self.final_answer(self.llm, query, rag_content, messages=previous_messages):
            if isinstance(chunk, str):
                yield chunk
        
    def ircot_solve(self, query, previous_messages = [], max_iter = 3):
        previous_messages = prune_think_tag(previous_messages)

        return ircot_pipeline_agent(
            query=query,
            llm=self.llm,
            retriever=self.rag,
            final_answer_function=self.final_answer,
            previous_messages=previous_messages,
            max_steps=max_iter,
        )


    def solve(self, query, previous_messages = [], max_iter = 3):
        
        previous_messages = prune_think_tag(previous_messages)

        ignore_ids = set()
        
        messages = [
            {
                'role' : 'system',
                'content': system_prompt
            },
        ]
        total_hop = 1
        
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
        intermediate_question = [query]

        rag_content = ''
        response = 'Anw cut'
        
        yield '\n<think>\n'
        
        for i in range(max_iter):
            # remove prev
            new_rag_content, new_ids =  self.rag(intermediate_question, ignore_ids = ignore_ids)           
            rag_content += new_rag_content
            ignore_ids.update(new_ids)
            
            content = f"Here is the provided content: \n\n {rag_content}. Answer the followning question \n\n<question>\n\n{query}\n\n</question>"

            messages.append(
                {
                    'role' : 'user',
                    'content': content
                }
            )

            yield f'\n================= {i + 1} ================\n\n'
            response = ''
            stream_response = self.llm.stream(messages)
            for chunk in stream_response:
                if isinstance(chunk, str):
                    response += chunk
                    yield chunk
            


            decision, detail = self.get_decision(response)
            
            messages.append(
                {
                    'role':'assistant',
                    'content': response
                }
            )
            

            intermediate_question = detail
            
            # Break condition
            if decision or len(intermediate_question) == 0:
                break
        
            total_hop += len(detail)
            yield f'\n\n========= Step {i + 1} end with {len(detail)} Sub-question ========\n'
            
        
        yield f'\n\n========== Total process end with {total_hop} hops =============\n\n'  
        
        yield '</think>\n'  


        for chunk in self.final_answer(self.llm, query, rag_content, messages=previous_messages):
            if isinstance(chunk, str):
                yield chunk
                
    
    def stream(self, messages, model_name: str = None):
        
        rag_strategy = 'default'
        
        if model_name.count(':') == 1:
            model_name, rag_strategy = model_name.split(':')
        
        if model_name is not None:
            self.init_llm(model_name)
        
        query = messages[-1]['content']
        messages = messages[:-1]
        
        print(f"RAG Strategy: {rag_strategy}")
        
        if rag_strategy == 'basic':
            return self.basic_solve(query, previous_messages=messages)
        elif rag_strategy == 'ircot':
            return self.ircot_solve(query, previous_messages=messages)
        
        return self.solve(query, messages)

        
if __name__ == "__main__":
    chat = Chat(model_name='qwen/qwen2.5-7b-instruct')
    
    query = "Who is the CEO of NVIDIA?"
    
    messages = [

        {
            'role': 'user',
            'content': query
        }
    ]
    
    for response in chat.stream(messages, model_name='qwen/qwen2.5-7b-instruct'):
        print(response, end='', flush=True)