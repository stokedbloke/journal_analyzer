# test.py
print("Testing imports...")

import sys
print(sys.version)
print(sys.executable)

print("1. Testing NumPy...")
import numpy as np
print(f"NumPy version: {np.__version__}")

print("\n2. Testing basic packages...")
import pandas as pd
from notion_client import Client
import nltk
print("Basic packages imported successfully")

print("\n3. Testing ML packages...")
import torch
print(f"PyTorch version: {torch.__version__}")

print("\n4. Testing transformers...")
from transformers import pipeline
print("Transformers imported successfully")

print("\nAll imports completed successfully!")