import openai
import google.generativeai as genai
import anthropic
from transformers import pipeline
from typing import Dict
from abc import ABC, abstractmethod


# Abstract base class for summarization
class SummarizationService(ABC):
    @abstractmethod
    def summarize(self, text: str) -> Dict:
        """Return a summary dictionary with a 'summary' key."""
        pass


# BART Summarizer using Hugging Face Transformers
class BartSummarizationService(SummarizationService):
    def __init__(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def summarize(self, text: str) -> Dict:
        result = self.summarizer(text, max_length=130, min_length=30, do_sample=False)
        return {"summary": result[0]["summary_text"].strip()}


# OpenAI GPT-based Summarizer (GPT-4/GPT-4o)
class OpenAISummarizationService(SummarizationService):
    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        openai.api_key = api_key
        self.model_name = model_name

    def summarize(self, text: str) -> Dict:
        prompt = f"Please summarize the following text concisely:\n\n{text}"
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return {"summary": response.choices[0].message.content.strip()}


# Google Gemini Summarizer (Gemini Pro)
class GeminiSummarizationService(SummarizationService):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-pro")

    def summarize(self, text: str) -> Dict:
        prompt = f"Summarize the following text concisely:\n\n{text}"
        response = self.model.generate_content(prompt)
        return {"summary": response.text.strip()}


# Anthropic Claude Summarizer (Claude 3 Opus by default)
class ClaudeSummarizationService(SummarizationService):
    def __init__(self, api_key: str, model_name: str = "claude-3-opus-20240229"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name

    def summarize(self, text: str) -> Dict:
        prompt = f"Summarize the following text concisely:\n\n{text}"
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=300,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return {"summary": response.content[0].text.strip()}
