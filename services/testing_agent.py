import sys
import io
from abc import ABC, abstractmethod
from typing import Dict

import openai
import google.generativeai as genai
import anthropic


# Base Interface
class TestingAgent(ABC):
    @abstractmethod
    def run_tests(self, code: str) -> Dict:
        pass


# 1. Local Testing Agent
class LocalTestingAgent(TestingAgent):
    def run_tests(self, code: str) -> Dict:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = captured_output = io.StringIO()
        sys.stderr = captured_error = io.StringIO()

        result = {
            "success": True,
            "output": "",
            "error": ""
        }

        try:
            exec(code, {})
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        result["output"] = captured_output.getvalue().strip()
        result["error"] += captured_error.getvalue().strip()

        return result


# 2. OpenAI Testing Agent
class OpenAITestingAgent(TestingAgent):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        openai.api_key = api_key
        self.model = model

    def run_tests(self, code: str) -> Dict:
        prompt = (
            "Review and run this code. Identify any issues and simulate output. "
            "Respond with the output or any errors.\n\n"
            f"{code}"
        )

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            "success": True,
            "output": response.choices[0].message.content.strip(),
            "error": ""
        }


# 3. Gemini Testing Agent
class GeminiTestingAgent(TestingAgent):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-pro")

    def run_tests(self, code: str) -> Dict:
        prompt = (
            "Simulate executing the following code and give the output or any errors.\n\n"
            f"{code}"
        )

        response = self.model.generate_content(prompt)
        return {
            "success": True,
            "output": response.text.strip(),
            "error": ""
        }


# 4. Claude Testing Agent
class ClaudeTestingAgent(TestingAgent):
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def run_tests(self, code: str) -> Dict:
        prompt = (
            "Simulate running the following Python code. Return any output or errors:\n\n"
            f"{code}"
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            "success": True,
            "output": response.content[0].text.strip(),
            "error": ""
        }
