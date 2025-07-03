import os
import openai
import anthropic
import google.generativeai as genai
from abc import ABC, abstractmethod
from typing import Dict
from datetime import datetime


class CodingAgent(ABC):
    @abstractmethod
    def generate_code(self, task: str) -> Dict:
        pass


def detect_language_from_first_line(content: str) -> str:
    first_line = content.strip().splitlines()[0].lower()
    if "python" in first_line:
        return "Python"
    elif "javascript" in first_line or "js" in first_line:
        return "JavaScript"
    elif "java" in first_line:
        return "Java"
    elif "html" in first_line:
        return "HTML"
    elif "c++" in first_line or "cpp" in first_line:
        return "C++"
    elif "c#" in first_line:
        return "C#"
    elif "typescript" in first_line:
        return "TypeScript"
    elif "go" in first_line:
        return "Go"
    else:
        return "Unknown"


def save_code_to_file(code: str, language: str, provider: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    language_ext = {
        "Python": "py",
        "JavaScript": "js",
        "Java": "java",
        "HTML": "html",
        "C++": "cpp",
        "C#": "cs",
        "TypeScript": "ts",
        "Go": "go"
    }
    ext = language_ext.get(language, "txt")
    filename = f"{provider.lower()}_{language.lower()}_{timestamp}.{ext}"
    output_dir = "generated_code"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)
    return file_path


class OpenAICodingAgent(CodingAgent):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        openai.api_key = api_key
        self.model = model

    def generate_code(self, task: str) -> Dict:
        prompt = (
            f"Generate clean and well-commented code to accomplish the following task:\n\n"
            f"{task}\n\n"
            f"Only include the code. Mention the programming language at the top."
        )
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content.strip()
        language = detect_language_from_first_line(content)
        file_path = save_code_to_file(content, language, "openai")
        return {"code": content, "language": language, "file_path": file_path}


class ClaudeCodingAgent(CodingAgent):
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate_code(self, task: str) -> Dict:
        prompt = (
            f"Generate clean and well-commented code to solve the following task:\n\n"
            f"{task}\n\n"
            f"Only include the code. Mention the programming language at the top."
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.content[0].text.strip()
        language = detect_language_from_first_line(content)
        file_path = save_code_to_file(content, language, "claude")
        return {"code": content, "language": language, "file_path": file_path}


class GeminiCodingAgent(CodingAgent):
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def generate_code(self, task: str) -> Dict:
        prompt = (
            f"Generate clean and well-commented code for the following task:\n\n"
            f"{task}\n\n"
            f"Only include the code. Mention the programming language at the top."
        )
        response = self.model.generate_content(prompt)
        content = response.text.strip()
        language = detect_language_from_first_line(content)
        file_path = save_code_to_file(content, language, "gemini")
        return {"code": content, "language": language, "file_path": file_path}


class GrokCodingAgent(CodingAgent):
    def __init__(self, api_key: str, model: str = "grok-1"):
        self.api_key = api_key
        self.model = model

    def generate_code(self, task: str) -> Dict:
        prompt = (
            f"Generate clean and well-commented code for the following task:\n\n"
            f"{task}\n\n"
            f"Only include the code. Mention the programming language at the top."
        )
        # Replace this stub with actual Grok API call when available
        content = "# Python\n# Example output from Grok\nprint('Hello from Grok!')"
        language = detect_language_from_first_line(content)
        file_path = save_code_to_file(content, language, "grok")
        return {"code": content, "language": language, "file_path": file_path}
