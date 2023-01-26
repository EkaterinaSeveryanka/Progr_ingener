from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

class Item(BaseModel):
    text: str

app = FastAPI()
t = pipeline("Image Transformer", model="facebook/deit-base-patch16-224")

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/Image Transformer/")
def predict(item: Item):
    return t(item.text)[0]
