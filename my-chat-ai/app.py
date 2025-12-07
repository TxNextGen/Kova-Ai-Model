from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()


print("Loading Nova AI model...")
model_name = "distilgpt2"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Model loaded!")

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"status": "Nova AI is running!", "model": "Nova AI v0.1"}

@app.post("/chat")
def chat(request: ChatRequest):
  
    inputs = tokenizer.encode(request.message, return_tensors="pt")
    outputs = model.generate(
        inputs, 
        max_length=100, 
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7
    )
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {
        "response": response_text,
        "model": "Nova AI v0.1"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)