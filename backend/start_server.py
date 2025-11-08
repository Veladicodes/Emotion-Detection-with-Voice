"""
Unified launcher to start backend (FastAPI) and frontend (Next.js) servers
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from threading import Thread


ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend" / "Voice-Emotion-Detector"

SEPARATOR = "=" * 70


def _print_header(message: str) -> None:
    print()
    print(SEPARATOR)
    print(message)
    print(SEPARATOR)
    print()


def _stream_output(process: subprocess.Popen, prefix: str) -> None:
    try:
        for line in iter(process.stdout.readline, b""):
            if not line:
                break
            decoded = line.decode("utf-8", errors="ignore").rstrip()
            print(f"[{prefix}] {decoded}")
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[{prefix}] log streaming error: {exc}")


def _check_directory(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} directory not found: {path}")


def main() -> None:
    _print_header("🎤 Starting Emotion Voice stack")

    _check_directory(BACKEND_DIR, "Backend")
    _check_directory(FRONTEND_DIR, "Frontend")

    python_exec = sys.executable

    backend_cmd = [
        python_exec,
        "-m",
        "uvicorn",
        "main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]

    frontend_cmd = ["npm", "run", "dev"]

    processes: list[tuple[str, subprocess.Popen]] = []

    def cleanup() -> None:
        print()
        _print_header("🛑 Shutting down services")
        for name, proc in processes:
            if proc.poll() is None:
                print(f"Stopping {name}...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("All services stopped.")

    def signal_handler(*_args) -> None:
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        print("Launching backend (FastAPI) on http://127.0.0.1:8000")
        backend_process = subprocess.Popen(
            backend_cmd,
            cwd=BACKEND_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        processes.append(("Backend", backend_process))
        Thread(target=_stream_output, args=(backend_process, "BACKEND"), daemon=True).start()

        time.sleep(2)

        print("Launching frontend (Next.js) on http://localhost:3000")
        frontend_process = subprocess.Popen(
            frontend_cmd,
            cwd=FRONTEND_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=sys.platform.startswith("win"),
        )
        processes.append(("Frontend", frontend_process))
        Thread(target=_stream_output, args=(frontend_process, "FRONTEND"), daemon=True).start()

        print()
        print("Backend and frontend started. Press Ctrl+C to stop.")
        print()

        while True:
            time.sleep(1)
            for name, proc in processes:
                if proc.poll() is not None:
                    raise RuntimeError(f"{name} exited with code {proc.returncode}")
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    main()

