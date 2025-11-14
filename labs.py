import os
import re
import subprocess
from pathlib import Path
from difflib import SequenceMatcher

# Import LLM engine
from llm_agent import get_llm_response


# ========================================================
# UTILS (FIXED & SAFE)
# ========================================================

def normalize_path(path_text: str):
    """
    Safely expands ~ and does NOT break absolute paths.
    No more 'home -> /home/venom/venom' bug.
    """
    path_text = path_text.strip()
    expanded = os.path.expanduser(path_text)
    return str(Path(expanded))


def fuzzy_find(target_name: str):
    """
    Fuzzy match any file or folder inside HOME.
    Matches partial names like 'prediction.py'.
    """
    target = target_name.lower().strip()
    home = str(Path.home())
    candidates = []

    for root, dirs, files in os.walk(home):
        for item in dirs + files:
            score = SequenceMatcher(None, target, item.lower()).ratio()
            if score >= 0.65:
                candidates.append((score, os.path.join(root, item)))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def run_command(cmd, cwd=None):
    """
    Safely executes any shell command (Python, C, C++, Java, Bash).
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"[ERROR] Failed to run command: {e}"


# ========================================================
# ---------- HALLUCINATION STRESS TEST -------------------
# ========================================================

def hallucination_test(user_query: str):
    """
    Multi-stage hallucination validation.
    """

    answer = get_llm_response(
        f"Answer the following STRICTLY based on verified facts:\n\n{user_query}"
    )

    confidence = get_llm_response(
        f"Rate your confidence (0-100) for this answer:\n\nQuery: {user_query}\nAnswer:{answer}"
    )

    hallucinations = get_llm_response(
        f"List uncertain or unverifiable claims:\n\nAnswer:\n{answer}"
    )

    return {
        "query": user_query,
        "answer": answer,
        "confidence": confidence,
        "possible_hallucinations": hallucinations
    }


# ========================================================
# ---------- FILE EXECUTION LAB (FIXED) ------------------
# ========================================================

def execute_file(filepath: str):
    """
    Main execution engine for: .py, .c, .cpp, .java, .sh
    ALWAYS runs file inside its own folder (CWD fix).
    """
    if not os.path.exists(filepath):
        return f"[ERROR] File not found: {filepath}"

    filepath = normalize_path(filepath)
    script_dir = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    ext = filepath.split(".")[-1].lower()

    # ---------------- PYTHON ----------------
    if ext == "py":
        return run_command(f"python '{filename}'", cwd=script_dir)

    # ---------------- C++ -------------------
    if ext == "cpp":
        exe = filepath.replace(".cpp", "")
        out = run_command(f"g++ '{filepath}' -o '{exe}' -std=c++17")
        if os.path.exists(exe):
            return run_command(f"./{os.path.basename(exe)}", cwd=script_dir)
        return out

    # ---------------- C ---------------------
    if ext == "c":
        exe = filepath.replace(".c", "")
        out = run_command(f"gcc '{filepath}' -o '{exe}'")
        if os.path.exists(exe):
            return run_command(f"./{os.path.basename(exe)}", cwd=script_dir)
        return out

    # ---------------- JAVA ------------------
    if ext == "java":
        out = run_command(f"javac '{filepath}'")
        classname = os.path.splitext(filename)[0]
        return run_command(f"java '{classname}'", cwd=script_dir)

    # ---------------- SHELL SCRIPT ----------
    if ext == "sh":
        return run_command(f"bash '{filename}'", cwd=script_dir)

    return f"[ERROR] Unsupported file type: {ext}"


# ========================================================
# ---------- NATURAL LANGUAGE RUNNER ---------------------
# ========================================================

def run_code_from_natural_input(cmd: str):
    """
    Understands:
    - run prediction.py
    - encoder decoder ke andar prediction.py ko run karo
    - ~/AI/encoder-decoder/prediction.py run karo
    - run /home/venom/AI/prediction.py
    """

    # FULL PATH MATCHING — fixed regex
    match = re.search(r'([\~/\w\-\./]+?\.(py|cpp|c|java|sh))', cmd)
    if not match:
        return "[ERROR] No runnable file detected."

    raw_path = match.group(1).strip()
    expanded = normalize_path(raw_path)

    # Case 1: User gave an absolute or ~ path
    if os.path.exists(expanded):
        return execute_file(expanded)

    # Case 2: Fuzzy match by filename
    filename_only = os.path.basename(raw_path)
    fuzzy_path = fuzzy_find(filename_only)

    if fuzzy_path:
        return execute_file(fuzzy_path)

    return f"[ERROR] Could not locate file: {raw_path}"


# ========================================================
# ---------- LAB WRAPPERS (FINAL) ------------------------
# ========================================================

def run_hallucination_lab(query: str):
    res = hallucination_test(query)
    return f"""
--- 🧪 Hallucination Test Results ---

Query:
{res['query']}

Answer:
{res['answer']}

Confidence:
{res['confidence']}

Possible Hallucinations:
{res['possible_hallucinations']}

--- End of Test ---
"""


# labs.py

import subprocess, threading, sys, time

def run_code_lab(cmd):
    """
    Runs user code safely with timeout, live output, and guaranteed return.
    """
    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        output = []
        def reader(stream):
            for line in iter(stream.readline, ''):
                print(line, end='')
                sys.stdout.flush()
                output.append(line)

        t1 = threading.Thread(target=reader, args=(process.stdout,))
        t2 = threading.Thread(target=reader, args=(process.stderr,))

        t1.start()
        t2.start()

        # Max execution time 20 seconds
        timeout = 20
        start = time.time()

        while process.poll() is None:
            if time.time() - start > timeout:
                process.kill()
                return "❌ Code execution timeout (20s)."
            time.sleep(0.1)

        t1.join()
        t2.join()

        return "".join(output) if output else "✅ Code executed (no output)."

    except Exception as e:
        return f"❌ Lab execution error: {e}"
