"""
Main entry point for the CBAS application backend.

This script is responsible for:
1.  Initializing and starting all background worker threads.
2.  Starting a web server using Bottle and Gevent-WebSocket to communicate
    with the Electron frontend.
3.  Exposing Python functions to be called from the frontend JavaScript.
"""

# --- Suppress the specific DeprecationWarning from pkg_resources ---
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

# Standard library imports
import os
import sys
import socket
import torch

# Local application imports
import eel
import workthreads

# The following modules are imported to ensure their @eel.expose decorators
# are registered with the Eel library upon startup.
import startup_page
import record_page
import label_train_page
import visualize_page


def find_available_port(start_port=8000, max_tries=100) -> int:
    """
    Finds an available network port to avoid conflicts.

    Args:
        start_port (int): The first port to check.
        max_tries (int): The number of subsequent ports to check.

    Returns:
        int: An available port number.

    Raises:
        IOError: If no free port is found within the given range.
    """
    for i in range(max_tries):
        port_to_try = start_port + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port_to_try))
            return port_to_try
        except OSError:
            continue  # Port is already in use
    raise IOError("No free ports found for Eel application.")


def main():
    """Initializes the backend server and waits for the Electron app to connect."""

    # GPU DIAGNOSTIC CODE 
    print("--- PyTorch GPU Diagnostics ---")
    try:
        is_available = torch.cuda.is_available()
        print(f"CUDA available: {is_available}")
        if is_available:
            print(f"Device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            print("PyTorch cannot find a CUDA-enabled GPU.")
            print("Possible reasons: NVIDIA drivers not installed, CUDA toolkit mismatch, or unsupported GPU.")
    except Exception as e:
        print(f"An error occurred during GPU diagnostics: {e}")
    print("-----------------------------")

    # Eel needs to know where the 'frontend' folder is to find eel.js
    eel.init('frontend')

    # Start all background processing threads (encoding, training, etc.)
    workthreads.start_threads()

    # Find a free port for the web server to run on.
    port = find_available_port()

    print(f"Eel server starting on http://localhost:{port}")
    print("This server will wait for the Electron GUI to connect.")

    try:
        # Start the Eel server.
        # This is the final, correct configuration for a decoupled Electron app.
        eel.start(
            'index.html',     # A required placeholder; the page is not actually opened by this call.
            mode=None,        # CRITICAL: This tells Eel NOT to launch any browser or window.
            host='localhost', # Listen only on the local machine.
            port=port,        # The port the server will run on.
            block=True        # CRITICAL: This keeps the Python script alive until the app is closed.
        )
    except (SystemExit, MemoryError, KeyboardInterrupt):
        # This block will be executed when the Electron app closes, which kills this process.
        print("Shutdown signal received, Python process is terminating.")
    finally:
        # Ensure worker threads are stopped cleanly on exit.
        workthreads.stop_threads()


if __name__ == "__main__":
    # This ensures the main() function is called only when the script is executed directly.
    main()