from __future__ import annotations

import ast
import json
import os
import re
import selectors
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

try:
    import resource  # POSIX only
except ImportError:
    resource = None  # type: ignore

# -------------------------------- Helpers -------------------------------- #
def _to_str(x) -> str:
    """
    Canonicalise any JSON‑serialisable test‑case payload to a single
    newline‑delimited string suitable for feeding to `stdin`.
    """
    if isinstance(x, str):
        return x                     # already a single line
    if isinstance(x, (bytes, bytearray)):
        return x.decode()            # rare, but be safe
    if isinstance(x, list):
        # Recursively stringify nested lists and join with newlines
        return "\n".join(_to_str(e) for e in x)
    # Dicts / numbers / other scalars → JSON text
    return json.dumps(x, ensure_ascii=False)

def _normalize(text: str) -> str:
    """Trim trailing blank lines and per‑line trailing spaces."""
    return "\n".join(line.rstrip() for line in text.rstrip().splitlines())

def evaluate(challenge, raw_reply: str) -> Dict[str, Any]:
    executor = ProgramExecutor()
    program = executor._strip_fences(raw_reply)

    sample = challenge.extra or {}
    ver_raw = sample.get("verification_info") or sample.get("test_cases")

    # try JSON first, then Python‐literal
    try:
        if isinstance(ver_raw, str):
            try:
                ver_json = json.loads(ver_raw)
            except json.JSONDecodeError:
                ver_json = ast.literal_eval(ver_raw)
        else:
            ver_json = ver_raw
    except Exception as err:
        print(f"Failed to parse verification info: {err}")
        return {
            "observation": f"{err.args[0]}",
            "reward": 0,
            "info": {}
        }

    # extract test cases list
    cases = ver_json.get("test_cases")
    if not cases:
        return {
            "observation": "No public test cases available",
            "reward": 0,
            "info": {}
        }

    passed, total = 0, len(cases)
    details = []

    for i, case in enumerate(cases, start=1):
        ctype = case.get("type")
        raw_inp = case.get("input")
        raw_exp = case.get("output")

        if ctype == "stdin_stdout":
            inp = _to_str(raw_inp)
            if not inp.endswith("\n"):
                inp += "\n"
            exec_prog = program
            exp = _to_str(raw_exp)
        elif ctype == "function_call":
            fn = case.get("fn_name")
            # input is a list of args
            args = case.get("input", [])
            # wrap program with a call to fn(...) and print its result
            exec_prog = (
                program
                + "\n"
                + f"if __name__ == '__main__':\n"
                + f"    result = {fn}(*{args!r})\n"
                + "    print(result)"
            )
            inp = ""  # no stdin
            exp = _to_str(raw_exp[0]) if isinstance(raw_exp, list) and raw_exp else _to_str(raw_exp)
        else:
            print(f"Unknown test case type '{ctype}', skipping.")
            total -= 1
            continue

        try:
            out, err = executor.execute(exec_prog, inp)

        except subprocess.TimeoutExpired:
            out, err = "", "TIMEOUT"

        ok_run = not err.strip()
        out_norm = _normalize(out)
        exp_norm = _normalize(exp) if exp is not None else None
        correct = ok_run and (exp_norm is None or out_norm == exp_norm)
        if correct:
            passed += 1
            print(f"Test case {i} passed.")
        else:
            print(
                f"Test case {i} failed. Got: {out_norm!r}, Expected: {exp_norm!r}"
            )

        details.append(
            {
                "input": inp,
                "expected": exp_norm,
                "got": out_norm,
                "stderr": err.strip(),
                "passed": correct,
            }
        )

    score = 1.0 if passed == total else 0.0
    feedback = json.dumps(
        {"passed": passed, "total": total, "tests": details}, ensure_ascii=False
    )
    print(f"Evaluation completed with score: {score}")
    return {
        "observation": feedback,
        "reward": score,
        "info": {}
    }


# --------------------------------------------------------------------------- #
#                              Configuration                                  #
# --------------------------------------------------------------------------- #
DEFAULT_TIMEOUT_SEC   = 30           # wall‑clock limit
CPU_TIME_SEC          = 10           # hard CPU seconds (POSIX only)
MEM_LIMIT_BYTES       = 512 * 2**20  # 512 MiB
MAX_OUTPUT_BYTES      = 1_000_000    # 1 MB per stream before truncation
_FENCE_RE  = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)
_HAS_MAIN  = re.compile(r'if\s+__name__\s*==\s*[\'"]__main__[\'"]')


class ProgramExecutor:
    """
    A hardened, feature‑rich Python runner for *ABDUCTION* and *DEDUCTION* tasks.

    • Strips fences, autoruns solve(), two‑stage execution
    • Wallclock, CPU/time, memory, and output limits
    • Cleans up temp files & stray child processes
    """

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT_SEC,
        cpu_time: int = CPU_TIME_SEC,
        mem_bytes: int = MEM_LIMIT_BYTES,
        max_output: int = MAX_OUTPUT_BYTES,
    ) -> None:
        self.timeout     = timeout
        self.cpu_time    = cpu_time
        self.mem_bytes   = mem_bytes
        self.max_output  = max_output
        self._tmp_files: List[str] = []
        self._lock = threading.Lock()

    @staticmethod
    def _strip_fences(text: str) -> str:
        python_blocks = re.findall(r'```python\s*\n(.*?)```', text, re.DOTALL)
        if python_blocks:
            return python_blocks[-1].strip()
        
        code_blocks = re.findall(r'```(?:\w*)\s*\n(.*?)```', text, re.DOTALL)
        if code_blocks:
            return code_blocks[-1].strip()
        
        m = _FENCE_RE.search(text)
        return (m.group(1) if m else text).strip()

    @contextmanager
    def _tempfile(self, content: str):
        path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as fh:
                fh.write(content)
                path = fh.name
            with self._lock:
                self._tmp_files.append(path)
            yield path
        finally:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                finally:
                    with self._lock:
                        if path in self._tmp_files:
                            self._tmp_files.remove(path)

    def _posix_rlimits(self) -> None:
        """Best‑effort resource limits; never raise."""
        if resource is None:
            return

        def _try_set(res, vals):
            try:
                resource.setrlimit(res, vals)
            except Exception:
                pass

        # CPU seconds
        if self.cpu_time and hasattr(resource, "RLIMIT_CPU"):
            _try_set(resource.RLIMIT_CPU, (self.cpu_time, self.cpu_time + 1))

        # Memory: prefer RLIMIT_AS, fallback to RLIMIT_DATA
        if self.mem_bytes:
            if hasattr(resource, "RLIMIT_AS"):
                _try_set(resource.RLIMIT_AS, (self.mem_bytes, self.mem_bytes))
            elif hasattr(resource, "RLIMIT_DATA"):
                _try_set(resource.RLIMIT_DATA, (self.mem_bytes, self.mem_bytes))

        # Disable core dumps
        if hasattr(resource, "RLIMIT_CORE"):
            _try_set(resource.RLIMIT_CORE, (0, 0))

    def _run_once(
        self,
        script: str,
        stdin_data: str,
    ) -> Tuple[str, str]:
        """
        Low‑level runner with incremental read and hard caps.
        Returns (stdout, stderr)—each possibly truncated.
        """
        start = time.time()
        proc = subprocess.Popen(
            [sys.executable, script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,  # line‑buffered
            preexec_fn=(lambda: self._posix_rlimits()) if resource else None,
            close_fds=True,
        )

        # On POSIX, start a new process‑group so we can kill children too
        if os.name != "nt":
            try:
                os.setpgid(proc.pid, proc.pid)
            except Exception:
                pass

        # Feed stdin then close
        if proc.stdin:
            try:
                proc.stdin.write(stdin_data)
                proc.stdin.close()
            except BrokenPipeError:
                # Process terminated early, stdin pipe is broken
                pass

        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ)
        sel.register(proc.stderr, selectors.EVENT_READ)

        out_buf, err_buf = [], []
        out_size = err_size = 0
        truncated = False

        # Poll loop
        while True:
            if time.time() - start > self.timeout:
                truncated = True
                break
            for key, _ in sel.select(timeout=0.1):
                chunk = key.fileobj.readline()
                if not chunk:
                    sel.unregister(key.fileobj)
                    continue
                if key.fileobj is proc.stdout:
                    out_size += len(chunk.encode())
                    out_buf.append(chunk)
                    if out_size > self.max_output:
                        truncated = True
                        break
                else:
                    err_size += len(chunk.encode())
                    err_buf.append(chunk)
                    if err_size > self.max_output:
                        truncated = True
                        break
            if truncated:
                break
            if proc.poll() is not None and not sel.get_map():
                break

        # If truncated or timed out, kill the whole group
        if truncated and proc.poll() is None:
            try:
                if os.name != "nt":
                    os.killpg(proc.pid, 15)
                    time.sleep(0.2)
                    if proc.poll() is None:
                        os.killpg(proc.pid, 9)
                else:
                    proc.kill()
            except Exception:
                pass

        # Drain any remaining output
        try:
            rest_out, rest_err = proc.communicate(timeout=0.2)
            out_buf.append(rest_out or "")
            err_buf.append(rest_err or "")
        except Exception:
            pass

        out_text = "".join(out_buf)
        err_text = "".join(err_buf)
        if truncated:
            suffix = "\n…<truncated>"
            if len(out_text.encode()) > self.max_output:
                out_text = out_text[: self.max_output] + suffix
            if len(err_text.encode()) > self.max_output:
                err_text = err_text[: self.max_output] + suffix
            if time.time() - start > self.timeout:
                err_text += "\n[TIMEOUT]"

        return out_text, err_text

    def execute(self, raw_code: str, stdin: str | bytes = "") -> Tuple[str, str]:
        """
        Run *raw_code* with *stdin*.
        • Strips ``` fences.
        • If no output/error and a solve() exists without a guard, append a small runner and re‑run.
        """
        code = self._strip_fences(raw_code)

        def _need_auto(src: str, out: str, err: str) -> bool:
            return (
                not out.strip() and not err.strip()
                and "def solve" in src
                and not _HAS_MAIN.search(src)
            )

        with self._tempfile(code) as script:
            try:
                out, err = self._run_once(script, stdin)
            except subprocess.SubprocessError as e:
                return "", f"Execution error: {e}"

            if _need_auto(code, out, err):
                runner = (
                    "\n\nif __name__ == \"__main__\":\n"
                    "    res = solve()\n"
                    "    if res is not None:\n"
                    "        import sys\n"
                    "        if isinstance(res, (list, tuple)):\n"
                    "            print(*res)\n"
                    "        else:\n"
                    "            print(res)\n"
                )
                with self._tempfile(code + runner) as auto_script:
                    try:
                        out, err = self._run_once(auto_script, stdin)
                    except subprocess.SubprocessError as e:
                        return "", f"Execution error: {e}"

        return out, err

    def cleanup(self) -> None:
        """Remove any lingering temp files."""
        with self._lock:
            for f in list(self._tmp_files):
                try:
                    os.remove(f)
                except Exception:
                    pass
            self._tmp_files.clear()
