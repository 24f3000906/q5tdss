import os
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from google import genai
from google.genai import types


app = FastAPI()

# Required: enable CORS for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class CodeResponse(BaseModel):
    error: List[int]
    result: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int] = Field(default_factory=list)


def execute_python_code(code: str) -> dict:
    """
    Execute Python code and return its exact stdout,
    or the exact traceback if an error occurs.
    """
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code, {"__name__": "__main__"})
        output = sys.stdout.getvalue()
        return {
            "success": True,
            "output": output
        }

    except Exception:
        output = traceback.format_exc()
        return {
            "success": False,
            "output": output
        }

    finally:
        sys.stdout = old_stdout


def analyze_error_with_ai(code: str, error_traceback: str) -> List[int]:
    """
    Ask Gemini to identify the line number(s) responsible for the error.
    Uses structured output so the response is valid JSON.
    """
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return []

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are analyzing a Python program that failed.

Identify the exact line number(s) in the CODE that caused the error.

Rules:
- Return only line numbers that directly caused or revealed the error.
- Line numbers start at 1.
- If multiple lines are involved, return all relevant line numbers.
- Do not invent line numbers.

CODE:
{code}

TRACEBACK:
{error_traceback}
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ErrorAnalysis,
        ),
    )

    analysis = ErrorAnalysis.model_validate_json(response.text)
    return analysis.error_lines


@app.post("/code-interpreter", response_model=CodeResponse)
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    if execution["success"]:
        return CodeResponse(
            error=[],
            result=execution["output"]
        )

    error_lines = analyze_error_with_ai(
        code=request.code,
        error_traceback=execution["output"]
    )

    return CodeResponse(
        error=error_lines,
        result=execution["output"]
    )
