import os
from dotenv import load_dotenv
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Dict, Optional

# Summarization + Doc Generation
from services.summarization import (
    BartSummarizationService,
    OpenAISummarizationService,
    GeminiSummarizationService,
    ClaudeSummarizationService,
    SummarizationService,
)
from services.docbuilder import generate_project_doc

# Code Generation
from services.coding_agent import (
    OpenAICodingAgent,
    GeminiCodingAgent,
    ClaudeCodingAgent,
    GrokCodingAgent,
    CodingAgent,
)

# Code Testing
from services.testing_agent import (
    LocalTestingAgent,
    TestingAgent,
)

# Load environment
print("STARTING MAIN.PY - MULTI LLM VERSION")
load_dotenv()
app = FastAPI()

# --- Schemas ---
class SummarizationRequest(BaseModel):
    transcript: str
    api_key: Optional[str] = None

class SummarizationResponse(BaseModel):
    summary: str
    project_doc: str

class CodeGenerationRequest(BaseModel):
    task: str
    api_key: Optional[str] = None
    provider: Optional[str] = "openai"
    model: Optional[str] = None

class CodeGenerationResponse(BaseModel):
    code: str
    language: str
    file_path: str
    test_output: Optional[Dict] = None

class CodeTestRequest(BaseModel):
    code: str

class CodeTestResponse(BaseModel):
    success: bool
    output: str
    error: str


# --- Summarize + Generate Doc ---
@app.post("/summarize", response_model=SummarizationResponse)
def summarize(request: SummarizationRequest) -> Dict:
    summary_provider = os.getenv("SUMMARY_PROVIDER", "BART").upper()
    api_key = request.api_key

    if summary_provider == "GPT-4":
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        summarizer: SummarizationService = OpenAISummarizationService(api_key=api_key)
    elif summary_provider == "GEMINI":
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        summarizer: SummarizationService = GeminiSummarizationService(api_key=api_key)
    elif summary_provider == "CLAUDE":
        api_key = api_key or os.getenv("CLAUDE_API_KEY")
        summarizer: SummarizationService = ClaudeSummarizationService(api_key=api_key)
    else:
        summarizer: SummarizationService = BartSummarizationService()

    result = summarizer.summarize(request.transcript)
    summary_text = result["summary"]
    doc_path = generate_project_doc(summary_text)

    return {
        "summary": summary_text,
        "project_doc": f"Document saved at: {doc_path}"
    }


# --- Code Generation + Auto Test ---
@app.post("/generate-code", response_model=CodeGenerationResponse)
def generate_code(request: CodeGenerationRequest) -> Dict:
    provider = (request.provider or "openai").lower()
    model = request.model
    api_key = request.api_key

    if provider == "openai":
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        agent: CodingAgent = OpenAICodingAgent(api_key=api_key, model=model or "gpt-4")
    elif provider == "claude":
        api_key = api_key or os.getenv("CLAUDE_API_KEY")
        agent: CodingAgent = ClaudeCodingAgent(api_key=api_key, model=model or "claude-3-opus-20240229")
    elif provider == "gemini":
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        agent: CodingAgent = GeminiCodingAgent(api_key=api_key, model=model or "gemini-pro")
    elif provider == "grok":
        api_key = api_key or os.getenv("GROK_API_KEY")
        agent: CodingAgent = GrokCodingAgent(api_key=api_key, model=model or "grok-1")
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Generate code
    result = agent.generate_code(request.task)
    code = result["code"]

    # Run auto test
    tester: TestingAgent = LocalTestingAgent()
    test_result = tester.run_tests(code)

    return {
        "code": result["code"],
        "language": result["language"],
        "file_path": result["file_path"],
        "test_output": test_result
    }


# --- Manual Code Testing ---
@app.post("/test-code", response_model=CodeTestResponse)
def test_code(request: CodeTestRequest) -> Dict:
    tester: TestingAgent = LocalTestingAgent()
    return tester.run_tests(request.code)
