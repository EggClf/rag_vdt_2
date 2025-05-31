from llm import OpenAIWrapper


host = 'http://localhost:8910/v1'
model = 'qwen/qwen2.5-7b-instruct'
api_key = 'your_api_key_here'
client = OpenAIWrapper(
    model_name=model,
    host=host,
    api_key=api_key
)

request = [
    {
        "role": "user",
        "content": "Who is the president of the United States?"
    }
]

for chunk in client.stream(request):
    print(chunk, end='', flush=True)