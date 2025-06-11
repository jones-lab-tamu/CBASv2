# Installing CBAS from Source (Windows)

This guide provides step-by-step instructions for installing either the stable v2 or the beta v3 of the CBAS application on a Windows machine.

## 1. Install Primary Dependencies

These tools are required for **both** v2 and v3.

1.  **Git:** The version control system used to download the code.
    *   Download from: [https://git-scm.com/download/win](https://git-scm.com/download/win) and install using the default settings.

2.  **Python 3.10 or 3.11:** The core programming language for the backend.
    *   Download from: [https://www.python.org/downloads/](https://www.python.org/downloads/) (select "Windows installer (64-bit)").
    *   During installation, **it is crucial that you check the box that says "Add Python to PATH"**.

3.  **FFmpeg:** A powerful tool for video processing used by CBAS for recording.
    *   Download the "essentials" build from: [https://gyan.dev/ffmpeg/builds/](https://gyan.dev/ffmpeg/builds/).
    *   Unzip the file and move the resulting folder to a permanent location (e.g., `C:\`).
    *   Add its `bin` folder to your Windows PATH environment variable (e.g., `C:\ffmpeg-7.0-essentials_build\bin`).

4.  **VLC Media Player (Optional):** Used for the "Live View" feature.
    *   Download from: [https://www.videolan.org/vlc/](https://www.videolan.org/).

## 2. Install Node.js (Required for v3 Beta)

If you intend to install the v3 Beta, you must also install **Node.js**.

*   Download the LTS version from: [https://nodejs.org/](https://nodejs.org/).
*   Run the installer with default settings.

## 3. Install and Run CBAS

1.  **Open Command Prompt** and navigate to where you want to store the project.
    ```
    cd C:\Users\YourName\Documents
    ```

2.  **Clone the Repository:** This downloads the CBAS source code.
    ```
    git clone https://github.com/jones-lab-tamu/CBAS.git
    ```

3.  **Navigate into the Project Folder:**
    ```
    cd CBAS
    ```

4.  **(IMPORTANT) Choose Which Version to Install and Run**

    The next steps depend entirely on which version you want to use.

    ---
    ### Option A: Install and Run v3 (BETA - Desktop App)

    a. **Switch to the `main` branch:**
        ```
        git checkout main
        ```

    b. **Create and Activate a Python Virtual Environment:**
        ```
        python -m venv venv
		```
		```
        .\venv\Scripts\activate
        ```

    c. **Install Python Dependencies:**
        ```
        pip install -r requirements.txt
        ```

    d. **Install PyTorch for GPU (Required):** This step installs the correct version of PyTorch needed for modern NVIDIA GPUs (RTX 20-series and newer).
        ```
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```

    e. **Install Node.js Dependencies:**
        ```
        npm install
        ```

    f. **Run the Application:**
        ```
        npm start
        ```

    ---
    ### Option B: Install and Run v2 (Stable - Web Browser App)
    
    *Note: v2 requires an older CUDA toolkit and Visual Studio for some dependencies. This setup may conflict with the v3 installation if performed in the same environment.*

    a. **Switch to the `v2-stable` branch:**
        ```
        git checkout v2-stable
        ```

    b. **Create and Activate a Python Virtual Environment:** (It is recommended to use a different folder or environment name if you also have v3).
        ```
        python -m venv venv-v2
		```
		```
        .\venv-v2\Scripts\activate
        ```

    c. **Install General Python Dependencies:**
        ```
        pip install -r requirements.txt
        ```
    
    d. **Install PyTorch for CUDA 11.8:**
        ```
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```
        
    e. **Install Visual Studio Build Tools:**
       *   Download from: [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
       *   Run the installer. In the workloads section, check the box for **"Desktop development with C++"** and click "Install".

    f. **Run the Application:**
        ```
        python backend/app.py
        ```
    ---

## 5. Updating CBAS


To update your local copy of CBAS to the latest version, follow these steps:

1.  **Open Command Prompt** and navigate to your `CBAS` project folder.
    ```
    cd C:\Path\To\Your\CBAS
    ```

2.  **(For v3 users) Activate your virtual environment:**
    ```
    .\venv\Scripts\activate
    ```

3.  **Make sure you are on the correct branch.**
    *   For the v3 Desktop App, run: `git checkout main`
    *   For the v2 Web App, run: `git checkout v2-stable`

4.  **Pull the latest code from GitHub:**
    ```
    git pull
    ```

5.  **Update your dependencies.** This step is crucial to install any new tools the updated code requires.

    *   **If you are using v3 (the `main` branch):**
        ```
        # Update Python packages
        pip install -r requirements.txt
        
        # Update Node.js packages (like Electron)
        npm install
        ```

    *   **If you are using v2 (the `v2-stable` branch):**
        ```
        # Update Python packages
        pip install -r requirements.txt
        
        # You may also need to re-run the specific PyTorch install if it has changed
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```