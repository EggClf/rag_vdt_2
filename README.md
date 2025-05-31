# Multihop Backend API

A configurable streaming chat API with RAG capabilities for multihop question answering.

## Quick Start

### Environment Setup

Set up your API keys at the beginning:

```bash
export NVIDIA_API_KEY=...
```

Or put it in .env file

### Installation

Install the LLM library:

```bash
git clone https://github.com/hung20gg/llm.git
pip install -r requirements.txt
```

### Running the Server

#### Using run.py (Recommended)

The `run.py` script provides a flexible way to start the server with custom configurations:

```bash
# Basic usage with default settings
python run.py

# Custom model and device
python run.py --model-name "meta/llama-3.1-8b-instruct" --device cpu

# Custom collection and Qdrant settings
python run.py --collection "demo_multihop" --qdrant-host "http://localhost:6333"

# Development mode with auto-reload
python run.py --reload --debug --log-level debug

# Production mode with multiple workers
python run.py --workers 4 --host 0.0.0.0 --port 8910
```

#### Available Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-name` | `qwen/qwen2.5-7b-instruct` | Model name for the chat agent |
| `--collection` | `demo_multihop` | Qdrant collection name |
| `--qdrant-host` | `http://localhost:6333` | Qdrant host URL |
| `--device` | `cuda` | Device for inference (`cuda` or `cpu`) |
| `--host` | `0.0.0.0` | Host to bind the server |
| `--port` | `8000` | Port to bind the server |
| `--workers` | `1` | Number of worker processes |
| `--log-level` | `info` | Logging level |
| `--reload` | `False` | Enable auto-reload for development |
| `--debug` | `False` | Enable debug mode |

#### Direct API Usage

You can also run the streaming API directly:

```bash
# Direct streaming API
python api/streaming.py

# With uvicorn
uvicorn api.streaming:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### Health Check

```bash
curl http://localhost:8910/
curl http://localhost:8910/health
```

### Chat Completions

```bash
curl -X POST http://localhost:8910/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "stream": true
  }'
```

## Architecture

- **Chat Agent**: Configurable RAG-enabled chat with multihop reasoning
- **Streaming API**: OpenAI-compatible streaming endpoint
- **Qdrant Integration**: Vector database for document retrieval
- **Flexible Models**: Support for various LLM providers (OpenAI, Nvidia, local models)

## Configuration Examples

See `examples.sh` for more usage examples.

## Development

For development with auto-reload:

```bash
python run.py --reload --debug --log-level debug
```

## Production

For production deployment:

```bash
python run.py --workers 4 --log-level warning --host 0.0.0.0 --port 8080
```