from llm import OpenAIWrapper


host = 'http://localhost:8910/v1'
model = 'qwen/qwen2.5-7b-instruct:ircot'
api_key = 'your_api_key_here'
client = OpenAIWrapper(
    model_name=model,
    host=host,
    api_key=api_key
)

request = [
    {
        "role": "user",
        "content": """Does the article from Fortune suggest that the Federal Reserve’s interest rate hikes are a response to past
conditions, such as booming home prices, while The Sydney Morning Herald article indicates that the
Federal Reserve’s future interest rate decisions will be based on incoming economic data?"""
    }
]

for chunk in client.stream(request):
    print(chunk, end='', flush=True)