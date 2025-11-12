import os
from typing import List, Dict


try:
    from openai import types
except Exception:
    class _Schema:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _FunctionDeclaration:
        def __init__(self, name: str, description: str, parameters):
            self.name = name
            self.description = description
            self.parameters = parameters

    class _Tool:
        def __init__(self, function_declarations: List):
            self.function_declarations = function_declarations

    class _Type:
        OBJECT = "object"
        STRING = "string"

    types = type("types", (), {
        "Schema": _Schema,
        "FunctionDeclaration": _FunctionDeclaration,
        "Tool": _Tool,
        "Type": _Type,
    })()

# Описание схемы функции, которое ожидает main.py (schema_get_files_info)
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

def get_files_info(directory: str = ".") -> Dict:

    base = os.path.abspath(os.getcwd())
    target = os.path.abspath(os.path.join(base, directory or "."))

    # Защита от выхода за пределы рабочей директории
    if not target.startswith(base):
        return {"error": "Requested directory is outside the working directory."}

    results = []
    try:
        for entry in os.scandir(target):
            if entry.is_file():
                rel = os.path.relpath(entry.path, base)
                results.append({"path": rel, "size": entry.stat().st_size})
    except FileNotFoundError:
        return {"error": "Directory not found."}
    except PermissionError:
        return {"error": "Permission denied."}

    return {"files": results}