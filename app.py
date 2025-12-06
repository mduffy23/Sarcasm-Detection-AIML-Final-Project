# Packages
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define input and output
class SarcasmInput(BaseModel):
    Statement: str

class PredictionOutput(BaseModel):
    Sarcastic: bool
    sarcastic_probability: float

# Load in fine tuned RoBerta
model_path = ''

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu" # Would be nice to have cuda
model.to(device)

# Define function for lexical feature generation on user input
intensifiers = {"literally", "absolutely", "totally", "completely", "seriously"}
irony = {"yeah right", "sure", "as if", "i bet", "no way", "oh great", "oh yeah"}
sarcasm_punctuation = {"!","!!","!?","?!"}

def count_elongated(token):
    return 1 if re.search(r"(.)\1\1+", token) else 0

def lexical_features_one(text):
    text = str(text)
    text_lower = text.lower()
    tokens = re.findall(r'\w+|\S', text)

    intensifier_count = sum(t.lower() in intensifiers for t in tokens)
    irony_present = 1 if any(phrase in text_lower for phrase in irony) else 0
    punctuation_count = sum(text.count(p) for p in sarcasm_punctuation)
    cap_tokens = sum(1 for t in tokens if len(t) > 2 and t.isupper())
    cap_ratio = cap_tokens / max(len(tokens), 1)
    elongated_count = sum(count_elongated(t) for t in tokens)

    return [intensifier_count, irony_present, punctuation_count, cap_ratio, elongated_count]

# Predict sarcasm from natural language
def predict_sarcasm(model, text, tokenizer=tokenizer, device=device):
    
    # Tokenize
    encodings = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    
    # Lexical features
    lexical_features = torch.tensor([lexical_features_one(t) for t in text], dtype=torch.float32)
    
    # Move to device
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    lexical_features = lexical_features.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, lexical_features=lexical_features)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    
    return {"predicted_label": int(preds[0]), "probabilities": probs[0]}

# FastAPI
## Create FastAPI instance
app = FastAPI()

## Define prediction endpoint
@app.post("/predict", response_model = PredictionOutput)
def predict(data: SarcasmInput):
    statement = SarcasmInput.Statement

    prediction = predict_sarcasm(statement)
    
    is_sarcastic = prediction['predicted_label'] == 1
    sarcasm_probability = prediction['probabilities'][1]

    return PredictionOutput(is_sarcastic, sarcasm_probability)

# Mount Frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")