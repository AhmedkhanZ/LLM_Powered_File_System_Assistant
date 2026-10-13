
from pathlib import Path
from datetime import datetime, timezone
import json
import os

from pypdf import PdfReader
from docx import Document
from openai import OpenAI
from dotenv import load_dotenv


# ================== CONFIGURATION ==================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

# Load .env from the same directory as this script
load_dotenv(BASE_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY is missing. Configure it in your .env file."
    )

MODEL_NAME = "openai/gpt-4o-mini"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Create the output directory if needed
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ================== PATH VALIDATION ==================

def get_safe_path(filepath: str) -> Path:
    """Allow file operations only inside the project directory."""

    path = (BASE_DIR / filepath).resolve()

    try:
        path.relative_to(BASE_DIR)
    except ValueError:
        raise ValueError(
            "Access denied: path must remain inside the project directory."
        )

    return path


# ================== TOOL 1: READ FILE ==================

def read_file(filepath: str) -> dict:
    try:
        path = get_safe_path(filepath)

        if not path.is_file():
            return {
                "success": False,
                "error": "File not found"
            }

        extension = path.suffix.lower()

        if extension == ".txt":
            content = path.read_text(encoding="utf-8")

        elif extension == ".pdf":
            reader = PdfReader(str(path))
            content = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        elif extension == ".docx":
            document = Document(str(path))
            content = "\n".join(
                paragraph.text
                for paragraph in document.paragraphs
            )

        else:
            return {
                "success": False,
                "error": "Unsupported file type"
            }

        stat = path.stat()

        metadata = {
            "filename": path.name,
            "extension": extension,
            "size_bytes": stat.st_size,
            "modified_date": datetime.fromtimestamp(
                stat.st_mtime,
                tz=timezone.utc
            ).isoformat(),
            "character_count": len(content)
        }

        return {
            "success": True,
            "content": content,
            "metadata": metadata,
            "error": None
        }

    except Exception as error:
        return {
            "success": False,
            "error": f"Failed to read file: {error}"
        }


# ================== TOOL 2: LIST FILES ==================

def list_files(directory: str, extension: str = None) -> list:
    try:
        path = get_safe_path(directory)

        if not path.is_dir():
            raise NotADirectoryError(
                f"Directory not found: {directory}"
            )

        if extension is not None:
            extension = extension.lower()

            if not extension.startswith("."):
                extension = "." + extension

        results = []

        for entry in path.iterdir():
            if not entry.is_file():
                continue

            if extension is not None:
                if entry.suffix.lower() != extension:
                    continue

            stat = entry.stat()

            results.append({
                "name": entry.name,
                "size": stat.st_size,
                "modified_date": datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=timezone.utc
                ).isoformat()
            })

        return results

    except Exception as error:
        print(f"Failed to list files: {error}")
        return []


# ================== TOOL 3: WRITE FILE ==================

def write_file(filepath: str, content: str) -> dict:
    try:
        path = get_safe_path(filepath)

        # Prevent overwriting an existing file
        if path.exists():
            return {
                "success": False,
                "filepath": str(path),
                "error": "File already exists; refusing to overwrite."
            }

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        return {
            "success": True,
            "filepath": str(path),
            "error": None
        }

    except Exception as error:
        return {
            "success": False,
            "filepath": filepath,
            "error": str(error)
        }


# ================== TOOL 4: SEARCH IN FILE ==================

def search_in_file(filepath: str, keyword: str) -> dict:
    try:
        if not keyword or not keyword.strip():
            return {
                "success": False,
                "matches": [],
                "match_count": 0,
                "error": "Keyword cannot be empty"
            }

        result = read_file(filepath)

        if not result["success"]:
            return {
                "success": False,
                "matches": [],
                "match_count": 0,
                "error": result["error"]
            }

        content = result["content"]
        search_text = content.lower()
        search_keyword = keyword.lower()

        matches = []
        start = 0
        context_size = len(keyword) + 10

        while True:
            index = search_text.find(search_keyword, start)

            if index == -1:
                break

            context_start = max(0, index - context_size)
            context_end = min(
                len(content),
                index + len(keyword) + context_size
            )

            matches.append({
                "keyword": keyword,
                "position": index,
                "context": content[context_start:context_end]
            })

            start = index + len(keyword)

        return {
            "success": True,
            "filepath": filepath,
            "keyword": keyword,
            "matches": matches,
            "match_count": len(matches),
            "error": None
        }

    except Exception as error:
        return {
            "success": False,
            "filepath": filepath,
            "matches": [],
            "match_count": 0,
            "error": str(error)
        }


# ================== TOOL REGISTRY ==================

TOOLS = {
    "read_file": read_file,
    "list_files": list_files,
    "write_file": write_file,
    "search_in_file": search_in_file
}


# ================== TOOL SCHEMAS ==================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a TXT, PDF, or DOCX file and return its "
                "extracted content and metadata."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path relative to the project directory."
                    }
                },
                "required": ["filepath"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List files in a directory, optionally filtered by extension."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string"
                    },
                    "extension": {
                        "type": ["string", "null"]
                    }
                },
                "required": ["directory", "extension"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create a new text file inside the project directory. "
                "Do not overwrite existing files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string"
                    },
                    "content": {
                        "type": "string"
                    }
                },
                "required": ["filepath", "content"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_file",
            "description": (
                "Search a TXT, PDF, or DOCX file for a keyword "
                "and return matching context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string"
                    },
                    "keyword": {
                        "type": "string"
                    }
                },
                "required": ["filepath", "keyword"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]


# ================== ARGUMENT VALIDATION ==================

def validate_arguments(tool_name: str, arguments: dict) -> None:
    expected = {
        "read_file": {"filepath"},
        "list_files": {"directory", "extension"},
        "write_file": {"filepath", "content"},
        "search_in_file": {"filepath", "keyword"}
    }

    if tool_name not in expected:
        raise ValueError(f"Unknown tool: {tool_name}")

    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")

    if set(arguments) != expected[tool_name]:
        raise ValueError("Missing or unexpected tool arguments.")

    for key, value in arguments.items():
        if key == "extension" and value is None:
            continue

        if not isinstance(value, str):
            raise ValueError(
                f"Argument '{key}' must be a string."
            )


# ================== LLM ASSISTANT ==================

def ask_assistant(user_query: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a file assistant. Use the provided tools to "
                "read, list, write, and search files. Explain results "
                "clearly. Never claim a tool succeeded unless its result "
                "supports that claim. Use paths relative to the project "
                "directory. When creating a summary, read the source file "
                "first, write a concise and accurate summary to a new file "
                "inside the output directory, and report the result."
            )
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    max_iterations = 10

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=1024
        )

        assistant_message = response.choices[0].message

        # No tool call: return the assistant's final response
        if not assistant_message.tool_calls:
            return assistant_message.content or ""

        # Preserve the assistant message and its tool calls
        messages.append(
            assistant_message.model_dump(exclude_none=True)
        )

        # Execute each requested tool
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name

            try:
                arguments = json.loads(
                    tool_call.function.arguments
                )

                validate_arguments(tool_name, arguments)

                tool_function = TOOLS[tool_name]
                tool_result = tool_function(**arguments)

                result_json = json.dumps(
                    tool_result,
                    ensure_ascii=False,
                    default=str
                )

            except Exception as error:
                result_json = json.dumps({
                    "success": False,
                    "error": str(error)
                })

            # Send the tool result back to the model
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_json
            })

    return (
        "Stopped because the maximum tool-call "
        "iterations were reached."
    )


# ================== RUN THE ASSISTANT ==================

if __name__ == "__main__":
    print("File Assistant is ready.")

    query = (
        "Find resumes mentioning Python experience"
    )

    try:
        answer = ask_assistant(query)
        print("\nAssistant:", answer)

    except Exception as error:
        print(f"\nAssistant failed: {error}")
