"""
Serve fraud detection model as a REST API - NAIVE Version
"""

import pickle
from fastapi import fastapi
from pydantic import BaseModel, Field
from typing import Optional

# Load the trained model and encoder at startup
# This is loaded once when the server starts, not on every request
print("Loading model...")
with open("model/models.pkl", "rb") as f:
    model, encoder = pickle.load(f)
print("Successfully loaded model!")

# create FastAPI application