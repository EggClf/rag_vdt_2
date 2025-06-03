#!/usr/bin/env python3
"""
Run script for the multihop backend API server.
Allows configurable parameters for model, collection, device, and host settings.
"""

import argparse
import sys
import os
import uvicorn

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from api.streaming import create_app


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the multihop backend API server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen/qwen2.5-7b-instruct",
        help="Model name to use for the chat agent"
    )
    
    # Database configuration
    parser.add_argument(
        "--collection",
        type=str,
        default="demo_multihop",
        help="Qdrant collection name"
    )
    
    parser.add_argument(
        "--qdrant-host",
        type=str,
        default=os.getenv("QDRANT_HOST", "http://localhost:6333"),
        help="Qdrant host URL. Either a URL or :memory: for in-memory mode"
    )
    
    # Hardware configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for embedding model inference"
    )
    
    # Server configuration
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8910,
        help="Port to bind the server to"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of worker processes"
    )
    
    # Logging configuration
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Log level"
    )
    
    # Development options
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("🚀 Starting Multihop Backend API Server")
    print("=" * 50)
    print(f"Model: {args.model_name}")
    print(f"Collection: {args.collection}")
    print(f"Qdrant Host: {args.qdrant_host}")
    print(f"Device: {args.device}")
    print(f"Server: {args.host}:{args.port}")
    print("=" * 50)
    
    # Create the FastAPI app with the specified configuration
    app = create_app(
        model_name=args.model_name,
        collection_name=args.collection,
        qdrant_host=args.qdrant_host,
        device=args.device
    )
    
    # Configure uvicorn settings
    uvicorn_config = {
        "app": app,
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level,
        "workers": args.workers if not args.reload else 1,  # reload doesn't work with multiple workers
        "reload": args.reload,
    }
    
    # Add reload directories if in reload mode
    if args.reload:
        uvicorn_config["reload_dirs"] = [current_dir]
    
    try:
        # Run the server
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        print("\n👋 Server stopped gracefully")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()