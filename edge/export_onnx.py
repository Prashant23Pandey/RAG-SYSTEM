"""
ONNX Model Export for Edge/On-Device Deployment.

Exports embedding models and small SLMs to ONNX format for use in:
- Web browsers (transformers.js + WebGPU)
- Desktop apps (Tauri/Electron)
- Mobile devices (ONNX Runtime Mobile)

Usage:
    python export_onnx.py --model nomic-embed-text --output ./onnx/embeddings
    python export_onnx.py --model smollm2-135m --output ./onnx/slm
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("rag.onnx_export")


# =============================================================================
# ONNX Export Functions
# =============================================================================

def export_embedding_model(
    model_name: str = "nomic-ai/nomic-embed-text-v1",
    output_dir: str = "./onnx/embeddings",
    opset: int = 14,
    quantize: bool = True,
) -> dict:
    """Export embedding model to ONNX format."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer, AutoConfig
        from optimum.onnxruntime import ORTModelForFeatureExtraction
    except ImportError as e:
        return {"error": f"Missing dependencies: {e}. Install: pip install optimum[onnxruntime]"}

    print("\n" + "=" * 60)
    print("  Exporting Embedding Model to ONNX")
    print(f"  Model: {model_name}")
    print(f"  Output: {output_dir}")
    print("=" * 60 + "\n")

    os.makedirs(output_dir, exist_ok=True)

    print("[1/4] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("[2/4] Exporting to ONNX...")
    model = ORTModelForFeatureExtraction.from_pretrained(
        model_name,
        export=True,
        provider="CPUExecutionProvider",
    )

    print("[3/4] Saving model and tokenizer...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Create config for transformers.js compatibility
    config = {
        "model_type": "nomic_bert",
        "max_position_embeddings": 8192,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "vocab_size": 30522,
    }

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Quantize if requested
    if quantize:
        print("[4/4] Quantizing model...")
        try:
            from optimum.onnxruntime import ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig

            quantizer = ORTQuantizer.from_pretrained(model)
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)
            quantizer.quantize(
                save_dir=output_dir,
                quantization_config=qconfig,
            )
            print("  Quantization complete!")
        except Exception as e:
            print(f"  Quantization skipped: {e}")
    else:
        print("[4/4] Skipping quantization...")

    # Get file sizes
    total_size = 0
    for f in Path(output_dir).glob("*.onnx"):
        total_size += f.stat().st_size

    result = {
        "status": "success",
        "model": model_name,
        "output_dir": output_dir,
        "format": "onnx",
        "size_mb": round(total_size / (1024 * 1024), 2),
    }

    print(f"\nExport complete! Model size: {result['size_mb']} MB")
    return result


def export_slm_model(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    output_dir: str = "./onnx/slm",
    opset: int = 14,
    quantize: bool = True,
    max_length: int = 2048,
) -> dict:
    """Export small language model (SLM) to ONNX format."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from optimum.onnxruntime import ORTModelForCausalLM
    except ImportError as e:
        return {"error": f"Missing dependencies: {e}. Install: pip install optimum[onnxruntime]"}

    print("\n" + "=" * 60)
    print("  Exporting SLM to ONNX")
    print(f"  Model: {model_name}")
    print(f"  Output: {output_dir}")
    print("=" * 60 + "\n")

    os.makedirs(output_dir, exist_ok=True)

    print("[1/4] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("[2/4] Exporting to ONNX...")
    model = ORTModelForCausalLM.from_pretrained(
        model_name,
        export=True,
        provider="CPUExecutionProvider",
    )

    print("[3/4] Saving model and tokenizer...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Create config for transformers.js compatibility
    config = {
        "model_type": "llama",
        "max_position_embeddings": max_length,
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_hidden_layers": 22,
        "vocab_size": 32000,
    }

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Quantize if requested
    if quantize:
        print("[4/4] Quantizing model...")
        try:
            from optimum.onnxruntime import ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig

            quantizer = ORTQuantizer.from_pretrained(model)
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)
            quantizer.quantize(
                save_dir=output_dir,
                quantization_config=qconfig,
            )
            print("  Quantization complete!")
        except Exception as e:
            print(f"  Quantization skipped: {e}")
    else:
        print("[4/4] Skipping quantization...")

    # Get file sizes
    total_size = 0
    for f in Path(output_dir).glob("*.onnx"):
        total_size += f.stat().st_size

    result = {
        "status": "success",
        "model": model_name,
        "output_dir": output_dir,
        "format": "onnx",
        "size_mb": round(total_size / (1024 * 1024), 2),
    }

    print(f"\nExport complete! Model size: {result['size_mb']} MB")
    return result


def create_transformers_js_config(
    embedding_dir: str,
    slm_dir: str,
    output_path: str = "./edge/models.json",
) -> dict:
    """Create configuration file for transformers.js integration."""
    config = {
        "embedding": {
            "model_path": f"./onnx/{os.path.basename(embedding_dir)}",
            "tokenizer_path": f"./onnx/{os.path.basename(embedding_dir)}",
            "model_type": "nomic_bert",
            "max_length": 512,
        },
        "slm": {
            "model_path": f"./onnx/{os.path.basename(slm_dir)}",
            "tokenizer_path": f"./onnx/{os.path.basename(slm_dir)}",
            "model_type": "llama",
            "max_length": 2048,
        },
        "cloud_fallback": {
            "enabled": True,
            "endpoint": os.getenv("CLOUD_API_ENDPOINT", "https://api.example.com/rag"),
            "timeout_ms": 5000,
            "retry_count": 2,
        },
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to {output_path}")
    return config


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Export models to ONNX for edge deployment")
    subparsers = parser.add_subparsers(dest="command", help="Export command")

    # Embedding export
    p_emb = subparsers.add_parser("embeddings", help="Export embedding model")
    p_emb.add_argument("--model", default="nomic-ai/nomic-embed-text-v1")
    p_emb.add_argument("--output", default="./onnx/embeddings")
    p_emb.add_argument("--opset", type=int, default=14)
    p_emb.add_argument("--no-quantize", action="store_true")

    # SLM export
    p_slm = subparsers.add_parser("slm", help="Export small language model")
    p_slm.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p_slm.add_argument("--output", default="./onnx/slm")
    p_slm.add_argument("--opset", type=int, default=14)
    p_slm.add_argument("--no-quantize", action="store_true")
    p_slm.add_argument("--max-length", type=int, default=2048)

    # Config generation
    p_config = subparsers.add_parser("config", help="Generate transformers.js config")
    p_config.add_argument("--embedding-dir", default="./onnx/embeddings")
    p_config.add_argument("--slm-dir", default="./onnx/slm")
    p_config.add_argument("--output", default="./edge/models.json")

    args = parser.parse_args()

    if args.command == "embeddings":
        result = export_embedding_model(
            args.model, args.output, args.opset, not args.no_quantize
        )
    elif args.command == "slm":
        result = export_slm_model(
            args.model, args.output, args.opset, not args.no_quantize, args.max_length
        )
    elif args.command == "config":
        result = create_transformers_js_config(
            args.embedding_dir, args.slm_dir, args.output
        )
    else:
        parser.print_help()
        return

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
