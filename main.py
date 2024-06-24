from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
summarizer = pipeline("summarization", model="Falconsai/text_summarization")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ur")
class Text(BaseModel):
    text: str
class SummaryText(BaseModel):
    summary: str
class TranslationText(BaseModel):
    translation: str
@app.post("/summarize")
async def summarize_text(request: Text):
    summary = summarizer(request.text, max_length=700, min_length=100, do_sample=False)
    return {"summary": summary[0]['summary_text']}
@app.post("/translate")
async def translate_text(request: SummaryText):
    translation = translator(request.summary)
    return {"translation": translation[0]['translation_text']}