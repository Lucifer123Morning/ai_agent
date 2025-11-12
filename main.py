# python
import sys
import os
import importlib
import re
import shlex
import subprocess
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
    from dotenv import load_dotenv  # type: ignore

# Попытка динамически импортировать Google GenAI SDK и его types; при отсутствии создаём заглушки.
try:
    genai_pkg = importlib.import_module("google.genai")
    types = getattr(genai_pkg, "types", None) or importlib.import_module("google.genai.types")
    genai = genai_pkg
except Exception:
    genai = None

    class _Part:
        def __init__(self, text=None):
            self.text = text

        @classmethod
        def from_function_response(cls, name=None, response=None):
            p = cls(text=str(response))
            setattr(p, "function_response", SimpleNamespace(response=response))
            return p

    class _Content:
        def __init__(self, role=None, parts=None):
            self.role = role
            self.parts = parts or []

    class _GenerateContentConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _FunctionDeclaration:
        def __init__(self, name: str, description: str, parameters=None):
            self.name = name
            self.description = description
            self.parameters = parameters

    class _Tool:
        def __init__(self, function_declarations=None):
            self.function_declarations = function_declarations or []

    class _Schema:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _Type:
        OBJECT = "object"
        STRING = "string"

    class _TypesStub:
        Part = _Part
        Content = _Content
        GenerateContentConfig = _GenerateContentConfig
        FunctionDeclaration = _FunctionDeclaration
        Tool = _Tool
        Schema = _Schema
        Type = _Type

    types = _TypesStub()

# Динамический импорт load_dotenv и заглушка, если нет
try:
    _dotenv_mod = importlib.import_module("dotenv")
    load_dotenv = getattr(_dotenv_mod, "load_dotenv")
except Exception:
    def load_dotenv():
        return None

# Попытка импортировать существующую схему get_files_info (если есть)
try:
    from functions.get_files_info import schema_get_files_info
except Exception:
    schema_get_files_info = None

# Если schema_get_files_info отсутствует — создаём fallback
if schema_get_files_info is None:
    schema_get_files_info = types.FunctionDeclaration(
        name="get_files_info",
        description="Lists files in the specified directory along with their sizes, constrained to the working directory.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "directory": types.Schema(
                    type=types.Type.STRING,
                    description="The directory to list files from, relative to the working directory. If not provided, lists files in the working directory itself.",
                ),
            },
        ),
    )

# Дополнительные объявления функций
schema_get_file_content = types.FunctionDeclaration(
    name="get_file_content",
    description="Reads file contents relative to the working directory. Returns file text or an error if not accessible.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="Path to the file, relative to the working directory.",
            ),
        },
    ),
)

schema_run_python_file = types.FunctionDeclaration(
    name="run_python_file",
    description="Executes a Python file with optional command-line arguments. File path is relative to the working directory.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="Path to the Python file to execute, relative to the working directory.",
            ),
            "args": types.Schema(
                type=types.Type.STRING,
                description="Optional command-line arguments as a single string.",
            ),
        },
    ),
)

schema_write_file = types.FunctionDeclaration(
    name="write_file",
    description="Writes or overwrites a file at the given path with provided content. Path is relative to the working directory.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="Path to the file to write, relative to the working directory.",
            ),
            "content": types.Schema(
                type=types.Type.STRING,
                description="Content to write into the file.",
            ),
        },
    ),
)

# Обновлённое системное приглашение (включает все четыре операции)
system_prompt = """You are a helpful AI coding agent.

When a user asks a question or makes a request, make a function call plan. You can perform the following operations:
- List files and directories
- Read file contents
- Execute Python files with optional arguments
- Write or overwrite files

All paths you provide should be relative to the working directory. You do not need to specify the working directory in your function calls as it is automatically injected for security reasons.
"""

# Регистрируем все объявления в available_functions
available_functions = types.Tool(
    function_declarations=[
        d for d in (
            schema_get_files_info,
            schema_get_file_content,
            schema_run_python_file,
            schema_write_file,
        ) if d is not None
    ]
)

# --- Помощники и реальные реализации функций ---
def _safe_resolve(path, base):
    base_abs = os.path.abspath(base)
    target = os.path.abspath(os.path.join(base_abs, path))
    if not (target == base_abs or target.startswith(base_abs + os.sep)):
        raise ValueError("Path escapes working directory")
    return target

def _get_files_info(directory=".", working_directory="./calculator"):
    base = working_directory
    try:
        target = _safe_resolve(directory, base)
    except Exception as e:
        return f"Error: {e}"
    if not os.path.exists(target):
        return f"Error: directory not found: {directory}"
    entries = []
    try:
        for name in sorted(os.listdir(target)):
            p = os.path.join(target, name)
            if os.path.isfile(p):
                size = os.path.getsize(p)
                entries.append(f"{name} (file, {size} bytes)")
            elif os.path.isdir(p):
                entries.append(f"{name} (dir)")
            else:
                entries.append(f"{name} (other)")
    except Exception as e:
        return f"Error listing directory: {e}"
    return "\n".join(entries) if entries else "(empty)"

def _get_file_content(file_path, working_directory="./calculator"):
    try:
        target = _safe_resolve(file_path, working_directory)
    except Exception as e:
        return f"Error: {e}"
    if not os.path.exists(target) or not os.path.isfile(target):
        return f"Error: file not found: {file_path}"
    try:
        with open(target, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def _write_file(file_path, content, working_directory="./calculator"):
    try:
        target = _safe_resolve(file_path, working_directory)
    except Exception as e:
        return f"Error: {e}"
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content or "")
        return f"Wrote {len(content or '')} bytes to {file_path}"
    except Exception as e:
        return f"Error writing file: {e}"

def _run_python_file(file_path, args="", working_directory="./calculator"):
    try:
        target = _safe_resolve(file_path, working_directory)
    except Exception as e:
        return f"Error: {e}"
    if not os.path.exists(target) or not os.path.isfile(target):
        return f"Error: file not found: {file_path}"
    cmd = [os.sys.executable, target]
    if args:
        try:
            cmd += shlex.split(args)
        except Exception:
            cmd += [args]
    try:
        proc = subprocess.run(cmd, cwd=working_directory, capture_output=True, text=True, timeout=30)
        out = proc.stdout or ""
        err = proc.stderr or ""
        return f"Exit {proc.returncode}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
    except subprocess.TimeoutExpired:
        return "Error: execution timed out"
    except Exception as e:
        return f"Error running file: {e}"

# Основная функция, требуемая пользователем
def call_function(function_call_part, verbose=False):
    function_name = getattr(function_call_part, "name", None)
    function_args = dict(getattr(function_call_part, "args", {}) or {})

    if verbose:
        print(f"Calling function: {function_name}({function_args})")
    else:
        print(f" - Calling function: {function_name}")

    # Принудительно добавляем working_directory
    function_args["working_directory"] = "./calculator"

    # Отображение имени функции на реальную функцию
    dispatch = {
        "get_files_info": _get_files_info,
        "get_file_content": _get_file_content,
        "run_python_file": _run_python_file,
        "write_file": _write_file,
    }

    if function_name not in dispatch:
        err_response = {"error": f"Unknown function: {function_name}"}
        part_obj = None
        if hasattr(types.Part, "from_function_response"):
            try:
                part_obj = types.Part.from_function_response(name=function_name, response=err_response)
            except Exception:
                part_obj = None
        if part_obj is None:
            p = types.Part(text=str(err_response))
            setattr(p, "function_response", SimpleNamespace(response=err_response))
            part_obj = p
        return types.Content(role="tool", parts=[part_obj])

    try:
        result = dispatch[function_name](**function_args)
    except Exception as e:
        result = f"Error during execution: {e}"

    result_dict = {"result": result}
    part_obj = None
    if hasattr(types.Part, "from_function_response"):
        try:
            part_obj = types.Part.from_function_response(name=function_name, response=result_dict)
        except Exception:
            part_obj = None
    if part_obj is None:
        p = types.Part(text=str(result))
        setattr(p, "function_response", SimpleNamespace(response=result_dict))
        part_obj = p

    return types.Content(role="tool", parts=[part_obj])

# Заглушечный клиент для тестирования/локального запуска
class _ModelsStub:
    def generate_content(self, model, contents, config):
        # Найдём последнее текстовое сообщение в всей переписке (последний по времени)
        prompt_text = ""
        try:
            for content in reversed(contents):
                parts = getattr(content, "parts", []) or []
                for part in reversed(parts):
                    txt = getattr(part, "text", None)
                    if txt:
                        prompt_text = txt
                        break
                if prompt_text:
                    break
        except Exception:
            prompt_text = ""

        class Usage:
            prompt_token_count = 0
            candidates_token_count = 0

        class FunctionCall:
            def __init__(self, name, args):
                self.name = name
                self.args = args

        class Response:
            def __init__(self, txt, calls, candidates):
                self.usage_metadata = Usage()
                self.text = txt
                self.function_calls = calls
                self.candidates = candidates

        lowered = prompt_text.lower()

        # Если пользователь упомянул calculator — начнём с листинга рабочей директории
        if "calculator" in lowered:
            calls = [FunctionCall("get_files_info", {"directory": "."})]
            candidate_text = "Я хочу вызвать get_files_info для directory=."
            candidate = SimpleNamespace(content=types.Content(role="assistant", parts=[types.Part(text=candidate_text)]))
            return Response("Function call planned.", calls, [candidate])

        # Запрос списка файлов по ключевым словам
        if "pkg" in lowered:
            calls = [FunctionCall("get_files_info", {"directory": "pkg"})]
            candidate_text = "Я хочу вызвать get_files_info для directory=pkg"
            candidate = SimpleNamespace(content=types.Content(role="assistant", parts=[types.Part(text=candidate_text)]))
            return Response("Function call planned.", calls, [candidate])
        if "root" in lowered or "the root" in lowered or "root directory" in lowered:
            calls = [FunctionCall("get_files_info", {"directory": "."})]
            candidate_text = "Я хочу вызвать get_files_info для directory=."
            candidate = SimpleNamespace(content=types.Content(role="assistant", parts=[types.Part(text=candidate_text)]))
            return Response("Function call planned.", calls, [candidate])

        # Поиск упоминания пути/файла и цитированного контента
        file_match = re.search(r"([\/\w\.-]+\.\w+)", prompt_text)
        quoted_match = re.search(r"[\"']([^\"']+)[\"']", prompt_text)

        # Чтение файла
        if any(k in lowered for k in ("прочит", "читать", "read")) and file_match:
            file_path = file_match.group(1)
            calls = [FunctionCall("get_file_content", {"file_path": file_path})]
            candidate_text = f"Я хочу вызвать get_file_content для file_path={file_path}"
            candidate = SimpleNamespace(content=types.Content(role="assistant", parts=[types.Part(text=candidate_text)]))
            return Response("Function call planned.", calls, [candidate])

        # Запись в файл
        if any(k in lowered for k in ("напис", "запис", "write")) and file_match:
            file_path = file_match.group(1)
            content = quoted_match.group(1) if quoted_match else ""
            calls = [FunctionCall("write_file", {"file_path": file_path, "content": content})]
            candidate_text = f"Я хочу вызвать write_file для file_path={file_path}"
            candidate = SimpleNamespace(content=types.Content(role="assistant", parts=[types.Part(text=candidate_text)]))
            return Response("Function call planned.", calls, [candidate])

        # Запуск python-файла
        if any(k in lowered for k in ("запуст", "run")) and file_match:
            file_path = file_match.group(1)
            args_match = re.search(r"(?:с аргументами|with args|with arguments)\s+(.+)$", lowered)
            args = args_match.group(1).strip() if args_match else ""
            calls = [FunctionCall("run_python_file", {"file_path": file_path, "args": args})]
            candidate_text = f"Я хочу вызвать run_python_file для file_path={file_path}"
            candidate = SimpleNamespace(content=types.Content(role="assistant", parts=[types.Part(text=candidate_text)]))
            return Response("Function call planned.", calls, [candidate])

        # По умолчанию возврат текста (финальный ответ)
        candidate = SimpleNamespace(content=types.Content(role="assistant", parts=[types.Part(text="I'M JUST A ROBOT")] ))
        return Response("I'M JUST A ROBOT", [], [candidate])

class _ClientStub:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.models = _ModelsStub()

def main():
    load_dotenv()

    verbose = "--verbose" in sys.argv
    args = []
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            args.append(arg)

    if not args:
        print("AI Code Assistant")
        print('\\nUsage: python main.py \"your prompt here\" [--verbose]')
        print('Example: python main.py \"How do I fix the calculator?\"')
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY")

    if genai is not None:
        client = genai.Client(api_key=api_key)  # type: ignore
    else:
        client = _ClientStub(api_key=api_key)
        if verbose:
            print("Warning: google.genai not available, using stub client.")

    user_prompt = " ".join(args)

    if verbose:
        print(f"User prompt: {user_prompt}\n")

    messages = [
        types.Content(role="user", parts=[types.Part(text=user_prompt)]),
    ]

    config = types.GenerateContentConfig(
        tools=[available_functions],
        system_instruction=system_prompt
    )

    # Цикл многократных вызовов generate_content, максимум 20 итераций
    max_iters = 20
    for i in range(max_iters):
        if verbose:
            print(f"--- Iteration {i+1} ---")
        try:
            # Всегда отправляем весь messages
            response = client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=messages,
                config=config,
            )
        except Exception as e:
            print(f"Error calling generate_content: {e}")
            break

        # Обрабатываем candidates и добавляем их содержимое в messages
        candidates = getattr(response, "candidates", []) or []
        for cand in candidates:
            cand_content = getattr(cand, "content", None)
            if cand_content is not None:
                messages.append(cand_content)
                if verbose:
                    try:
                        text = getattr(cand_content.parts[0], "text", "")
                    except Exception:
                        text = str(cand_content)
                    print(f"Appended candidate content to messages: {text}")

        # Если модель вернула function_calls — вызываем их
        function_calls = getattr(response, "function_calls", None) or []
        if function_calls:
            for part in function_calls:
                function_call_result = call_function(part, verbose=verbose)
                # ожидаем parts[0].function_response.response
                try:
                    fr = function_call_result.parts[0].function_response.response
                except Exception:
                    raise RuntimeError("Function call did not return expected function_response structure")
                # создаём сообщение роли user из результата функции и добавляем в messages
                user_part = None
                if hasattr(types.Part, "from_function_response"):
                    try:
                        # fr сам уже словарь { "result": ... } или { "error": ... }
                        user_part = types.Part.from_function_response(name=part.name, response=fr)
                    except Exception:
                        user_part = None
                if user_part is None:
                    p = types.Part(text=str(fr))
                    setattr(p, "function_response", SimpleNamespace(response=fr))
                    user_part = p
                user_content = types.Content(role="user", parts=[user_part])
                messages.append(user_content)
                if verbose:
                    print(f"-> {fr}")

        # Если модель возвратила финальный текст — выводим и выходим
        final_text = getattr(response, "text", None)
        # Не завершаем, если есть запланированные вызовы функций или кандидаты — это не финальный ответ
        if final_text and not (function_calls or candidates):
            print("Final response:")
            print(final_text)
            break
        # иначе — переходим к следующей итерации (модель продолжит на основе обновлённого messages)
    else:
        # достигнут лимит итераций
        print("No final textual response after max iterations.")

if __name__ == "__main__":
    main()
