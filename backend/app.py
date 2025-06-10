"""
Main entry point for the CBAS application.

This script is responsible for:
1.  Finding an available network port for communication.
2.  Locating the Electron executable.
3.  Initializing the Eel library.
4.  Starting the background worker threads.
5.  Launching the Electron GUI process.
6.  Running the main application loop to keep the Python backend alive.
7.  Handling graceful shutdown of worker threads on exit.
"""

# Standard library imports
import os
import sys
import socket
import subprocess

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
    """Main function to initialize and start the Eel application."""
    
    # --- 1. Set up the Eel environment ---
    frontend_dir = "frontend"
    if not os.path.isdir(frontend_dir):
        # Handle case where backend is run from a different directory
        script_dir = os.path.dirname(__file__)
        frontend_dir = os.path.join(script_dir, '..', 'frontend')
        if not os.path.isdir(frontend_dir):
            print(f"Error: Cannot find 'frontend' directory.", file=sys.stderr)
            sys.exit(1)
            
    eel.init(frontend_dir)

    # --- 2. Find an available port ---
    try:
        port = find_available_port()
        print(f"Eel will run on http://localhost:{port}/index.html")
    except IOError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 3. Locate the Electron executable ---
    # This path assumes a standard `npm install` setup.
    electron_path = os.path.join("node_modules", "electron", "dist", "electron")
    if sys.platform == "win32":
        electron_path += ".exe"
    electron_path = os.path.abspath(electron_path)

    if not os.path.exists(electron_path):
        print("Warning: Electron executable not found.", file=sys.stderr)
        print(f"  - Searched at: {electron_path}", file=sys.stderr)
        print("  - Application will attempt to open in the default web browser instead.", file=sys.stderr)

    eel.browsers.set_path('electron', electron_path)

    # --- 4. Start Background Worker Threads ---
    print("Starting background worker threads...")
    workthreads.start_threads()

    # --- 5. Launch the Application GUI ---
    eel_options = {
        'mode': 'electron',
        'host': 'localhost',
        'port': port,
        'app_mode': True,  # Important for packaging with tools like PyInstaller
    }
    
    print("Launching GUI...")
    try:
        eel.start("index.html", options=eel_options, block=False, suppress_error=True)
    except Exception as e:
        # Fallback to browser mode if Electron fails for any reason
        print(f"Could not start in Electron mode: {e}", file=sys.stderr)
        print("Attempting to start in default browser instead...")
        try:
            eel.start("index.html", mode=None, port=port, block=False)
        except Exception as browser_e:
            print(f"Could not start Eel in any mode: {browser_e}", file=sys.stderr)
            workthreads.stop_threads()
            sys.exit(1)

    # --- 6. Run Main Application Loop ---
    # This loop keeps the Python script running while the GUI is open.
    # eel.sleep() allows Eel to process communications between Python and JS.
    try:
        while True:
            eel.sleep(1.0)
    except (KeyboardInterrupt, SystemExit, MemoryError):
        # A MemoryError in the main loop might mean the GUI process crashed.
        print("Main application loop interrupted. Shutting down...")
    finally:
        print("Cleaning up: Stopping all worker threads...")
        workthreads.stop_threads()
        print("Cleanup complete. Exiting.")


if __name__ == "__main__":
    main()