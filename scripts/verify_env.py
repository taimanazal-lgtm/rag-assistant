"""Verify that the environment is ready for the RAG project."""
import importlib
import sys


def check_gpu():
    import torch

    assert torch.cuda.is_available(), "CUDA/GPU not available"
    print(f"[OK] GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")


def check_imports():
    modules = ["pandas", "pyarrow", "sentence_transformers", "chromadb",
               "spacy", "bs4", "langdetect", "pytest", "dotenv"]
    for name in modules:
        importlib.import_module(name)
        print(f"[OK] import {name}")


def check_spacy_model():
    import spacy

    spacy.load("en_core_web_sm")
    print("[OK] spaCy model en_core_web_sm loaded")


def check_chromadb():
    import chromadb

    client = chromadb.Client()
    col = client.create_collection("verify_test")
    col.add(ids=["1"], documents=["hello world"], embeddings=[[0.1, 0.2, 0.3]])
    assert col.count() == 1
    print("[OK] ChromaDB in-memory test passed")


if __name__ == "__main__":
    print(f"Python {sys.version.split()[0]}")
    check_gpu()
    check_imports()
    check_spacy_model()
    check_chromadb()
    print("\nEnvironment verified successfully.")