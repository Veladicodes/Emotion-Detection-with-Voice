#!/usr/bin/env python3
"""
PHASE 5 TIER-0: Unified Server Launcher
Start both Backend and Frontend servers for Emotion Voice application
- Checks dependencies (including ffmpeg)
- Starts FastAPI backend with TIER-0 features
- Starts Next.js frontend
- Handles graceful shutdown
Press Ctrl+C to stop both servers
"""

import subprocess
import sys
import os
import time
import signal
import requests
from pathlib import Path
from threading import Thread

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(message):
    """Print a formatted header"""
    print()
    print(Colors.CYAN + "=" * 70 + Colors.ENDC)
    print(Colors.YELLOW + f"  {message}" + Colors.ENDC)
    print(Colors.CYAN + "=" * 70 + Colors.ENDC)
    print()

def print_colored(message, color=Colors.ENDC):
    """Print colored message"""
    print(color + message + Colors.ENDC)

def check_port_available(port):
    """Check if a port is available"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    return result != 0

def wait_for_backend(max_attempts=20):
    """Wait for backend to be ready"""
    print_colored("   Waiting for backend to initialize...", Colors.YELLOW)
    
    for i in range(max_attempts):
        try:
            response = requests.get("http://127.0.0.1:8000/health", timeout=2)
            if response.status_code == 200:
                print_colored("✅ Backend is healthy and ready!", Colors.GREEN)
                return True
        except requests.exceptions.RequestException:
            time.sleep(1)
    
    print_colored("⚠️  Backend may still be starting up...", Colors.YELLOW)
    return False

def stream_output(process, prefix, color):
    """Stream output from a process with colored prefix"""
    try:
        for line in iter(process.stdout.readline, b''):
            if line:
                decoded = line.decode('utf-8', errors='ignore').rstrip()
                print(f"{color}[{prefix}]{Colors.ENDC} {decoded}")
    except Exception as e:
        print_colored(f"Error streaming {prefix} output: {e}", Colors.RED)

def check_ffmpeg():
    """Check if ffmpeg is installed and accessible"""
    try:
        result = subprocess.run(["ffmpeg", "-version"], 
                              capture_output=True, 
                              timeout=5,
                              text=True)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0] if result.stdout else "unknown"
            print_colored(f"✅ FFmpeg found: {version}", Colors.GREEN)
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print_colored("⚠️  FFmpeg not found - WebM recording may not work", Colors.YELLOW)
    print_colored("   Install with: .\\INSTALL_FFMPEG_QUICK.ps1", Colors.YELLOW)
    
    # Add common ffmpeg paths to environment
    common_paths = [
        r"C:\ffmpeg\bin",
        r"C:\Program Files\ffmpeg\bin",
        os.path.expanduser("~/.local/bin")
    ]
    
    for path in common_paths:
        if os.path.exists(path) and path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
            print_colored(f"   Added {path} to PATH", Colors.YELLOW)
    
    return False


def check_tier0_dependencies(venv_python):
    """Check if TIER-0 dependencies are installed"""
    print_colored("\n🔍 Checking TIER-0 dependencies...", Colors.CYAN)
    
    required_packages = {
        'slowapi': '0.1.9',
        'pydantic': '2.0.0+',
        'pydantic-settings': '2.0.3',
        'psutil': '5.9.6',
        'python-dotenv': '1.0.0'
    }
    
    missing = []
    
    for package, version in required_packages.items():
        try:
            result = subprocess.run(
                [str(venv_python), "-c", f"import {package.replace('-', '_')}; print('OK')"],
                capture_output=True,
                timeout=5,
                text=True
            )
            if result.returncode == 0 and "OK" in result.stdout:
                print_colored(f"   ✅ {package}", Colors.GREEN)
            else:
                missing.append(package)
                print_colored(f"   ❌ {package} - not found", Colors.RED)
        except:
            missing.append(package)
            print_colored(f"   ❌ {package} - not found", Colors.RED)
    
    if missing:
        print_colored(f"\n⚠️  Missing {len(missing)} TIER-0 dependencies", Colors.YELLOW)
        print_colored("   Installing now...", Colors.YELLOW)
        
        try:
            install_cmd = [
                str(venv_python), "-m", "pip", "install",
                "slowapi==0.1.9",
                "pydantic>=2.0.0",
                "pydantic-settings==2.0.3",
                "psutil==5.9.6",
                "python-dotenv==1.0.0",
                "fastapi==0.104.1",
                "uvicorn[standard]==0.24.0"
            ]
            
            subprocess.run(install_cmd, check=True)
            print_colored("   ✅ Dependencies installed successfully!", Colors.GREEN)
        except subprocess.CalledProcessError as e:
            print_colored(f"   ❌ Failed to install dependencies: {e}", Colors.RED)
            print_colored("   Manual install: cd backend && pip install -r requirements.txt", Colors.YELLOW)
    else:
        print_colored("   ✅ All TIER-0 dependencies installed!", Colors.GREEN)


def main():
    # Get script directory
    ROOT_DIR = Path(__file__).parent
    BACKEND_DIR = ROOT_DIR / "backend"
    FRONTEND_DIR = ROOT_DIR / "frontend" / "Voice-Emotion-Detector"
    
    print_header("🎤 EMOTION VOICE - TIER-0 PRODUCTION")
    
    # Check if directories exist
    if not BACKEND_DIR.exists():
        print_colored(f"❌ Backend directory not found: {BACKEND_DIR}", Colors.RED)
        sys.exit(1)
    
    if not FRONTEND_DIR.exists():
        print_colored(f"❌ Frontend directory not found: {FRONTEND_DIR}", Colors.RED)
        sys.exit(1)
    
    # Check if virtual environment exists
    if sys.platform == "win32":
        venv_python = BACKEND_DIR / "venv" / "Scripts" / "python.exe"
    else:
        venv_python = BACKEND_DIR / "venv" / "bin" / "python"
    
    if not venv_python.exists():
        print_colored("❌ Backend virtual environment not found!", Colors.RED)
        print_colored("   Run: cd backend && python -m venv venv && pip install -r requirements.txt", Colors.YELLOW)
        sys.exit(1)
    
    # TIER-0: Check ffmpeg
    check_ffmpeg()
    
    # TIER-0: Check dependencies
    check_tier0_dependencies(venv_python)
    
    print()  # Add spacing
    
    # Check if node_modules exists
    node_modules = FRONTEND_DIR / "node_modules"
    if not node_modules.exists():
        print_colored("⚠️  Frontend node_modules not found. Installing dependencies...", Colors.YELLOW)
        try:
            subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, check=True)
            print_colored("✅ Frontend dependencies installed!", Colors.GREEN)
        except subprocess.CalledProcessError as e:
            print_colored(f"❌ Failed to install frontend dependencies: {e}", Colors.RED)
            sys.exit(1)
    else:
        print_colored("✅ Frontend dependencies ready", Colors.GREEN)
    
    # Check if ports are available
    if not check_port_available(8000):
        print_colored("⚠️  Port 8000 is already in use. Backend may already be running.", Colors.YELLOW)
        response = input("   Kill existing process and continue? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    if not check_port_available(3000):
        print_colored("⚠️  Port 3000 is already in use. Frontend may already be running.", Colors.YELLOW)
        response = input("   Kill existing process and continue? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Store process objects
    processes = []
    
    def cleanup():
        """Cleanup function to stop all servers"""
        print()
        print_header("🛑 STOPPING ALL SERVERS")
        
        for proc_name, proc in processes:
            if proc.poll() is None:  # Process is still running
                print_colored(f"   Stopping {proc_name}...", Colors.YELLOW)
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        
        print_colored("✅ All servers stopped!", Colors.GREEN)
        print()
    
    # Register signal handlers
    def signal_handler(sig, frame):
        cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start Backend
        print_header("🔧 STARTING BACKEND SERVER")
        print_colored(f"   Directory: {BACKEND_DIR}", Colors.CYAN)
        print_colored("   URL: http://127.0.0.1:8000", Colors.CYAN)
        
        backend_cmd = [
            str(venv_python),
            "-m",
            "uvicorn",
            "main:app",
            "--reload",
            "--host", "127.0.0.1",
            "--port", "8000"
        ]
        
        backend_process = subprocess.Popen(
            backend_cmd,
            cwd=BACKEND_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=False
        )
        
        processes.append(("Backend", backend_process))
        print_colored("✅ Backend process started", Colors.GREEN)
        
        # Start backend output thread
        backend_thread = Thread(
            target=stream_output,
            args=(backend_process, "BACKEND", Colors.BLUE),
            daemon=True
        )
        backend_thread.start()
        
        # Wait for backend to be ready
        time.sleep(3)
        wait_for_backend()
        
        # Start Frontend
        print_header("⚛️  STARTING FRONTEND SERVER")
        print_colored(f"   Directory: {FRONTEND_DIR}", Colors.CYAN)
        print_colored("   URL: http://localhost:3000", Colors.CYAN)
        
        frontend_cmd = ["npm", "run", "dev"]
        
        frontend_process = subprocess.Popen(
            frontend_cmd,
            cwd=FRONTEND_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=False,
            shell=True if sys.platform == "win32" else False
        )
        
        processes.append(("Frontend", frontend_process))
        print_colored("✅ Frontend process started", Colors.GREEN)
        
        # Start frontend output thread
        frontend_thread = Thread(
            target=stream_output,
            args=(frontend_process, "FRONTEND", Colors.GREEN),
            daemon=True
        )
        frontend_thread.start()
        
        # Display status
        print_header("✅ TIER-0 SERVICES STARTED")
        print()
        print_colored("   🔧 Backend API:    http://127.0.0.1:8000", Colors.CYAN)
        print_colored("   📚 API Docs:       http://127.0.0.1:8000/docs", Colors.CYAN)
        print_colored("   📊 Metrics:        http://127.0.0.1:8000/api/metrics", Colors.CYAN)
        print_colored("   🏥 Health:         http://127.0.0.1:8000/api/health/detailed", Colors.CYAN)
        print_colored("   ⚛️  Frontend UI:    http://localhost:3000", Colors.CYAN)
        print()
        print_colored("   🎤 Record voice and test emotion detection!", Colors.BOLD)
        print_colored("   📝 Trace logs:     backend/logs/inference_trace/", Colors.CYAN)
        print()
        print_colored("   Press Ctrl+C to stop all servers", Colors.YELLOW)
        print()
        print(Colors.CYAN + "=" * 70 + Colors.ENDC)
        print()
        print_colored("📝 Live logs (Backend in Blue, Frontend in Green):", Colors.BOLD)
        print()
        
        # Keep script running and monitor processes
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            backend_running = backend_process.poll() is None
            frontend_running = frontend_process.poll() is None
            
            if not backend_running:
                print_colored("❌ Backend process stopped unexpectedly!", Colors.RED)
                break
            
            if not frontend_running:
                print_colored("❌ Frontend process stopped unexpectedly!", Colors.RED)
                break
    
    except KeyboardInterrupt:
        print()
        print_colored("Received interrupt signal...", Colors.YELLOW)
    except Exception as e:
        print_colored(f"❌ Error: {e}", Colors.RED)
    finally:
        cleanup()

if __name__ == "__main__":
    main()

