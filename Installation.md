# Installing CBAS from Source (Windows)

This guide provides step-by-step instructions for installing either the stable v2 or the beta v3 of the CBAS application on a Windows machine.

## 1. Install Primary Dependencies

These tools are required for **both** v2 and v3.

1.  **Git:** The version control system used to download the code.
    *   Download from: [https://git-scm.com/download/win](https://git-scm.com/download/win) and install using the default settings.

2.  **Python 3.10:** The core programming language for the backend.
    *   Download from: [https://www.python.org/downloads/release/python-31011/](https://www.python.org/downloads/release/python-31011/) (select "Windows installer (64-bit)").
    *   During installation, **it is crucial that you check the box that says "Add Python 3.10 to PATH"**.

3.  **FFmpeg:** A powerful tool for video processing used by CBAS for recording.
    *   Download the "essentials" build from: [https://gyan.dev/ffmpeg/builds/](https://gyan.dev/ffmpeg/builds/).
    *   Unzip the file and move the resulting folder to a permanent location (e.g., `C:\`).
    *   Add its `bin` folder to your Windows PATH environment variable (e.g., `C:\ffmpeg-7.0-essentials_build\bin`).

4.  **VLC Media Player (Optional):** Used for the "Live View" feature.
    *   Download from: [https://www.videolan.org/vlc/](https://www.videolan.org/).

## 2. Install Version-Specific Dependencies

Your next step depends on which version of CBAS you intend to install.

*   **For v3 (BETA - Recommended Desktop App):** You must also install **Node.js**.
    *   Download the LTS version from: [https://nodejs.org/](https://nodejs.org/).
    *   Run the installer with default settings.

*   **For v2 (Stable - Web Browser App):** You must also install **Visual Studio Build Tools**.
    *   Download from: [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
    *   Run the installer. In the workloads section, check the box for **"Desktop development with C++"** and click "Install".

## 3. Install NVIDIA CUDA Toolkit (for GPU Acceleration)

This step is optional but **strongly recommended** if you have an NVIDIA GPU.

1.  **Download CUDA Toolkit 11.8:** [https://developer.nvidia.com/cuda-11-8-0-download-archive](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2.  Select: *Windows -> x86_64 -> 10 or 11 -> exe (local)*.
3.  Run the installer and choose the **"Express (Recommended)"** option.

## 4. Install and Run CBAS

1.  **Open Command Prompt** and navigate to where you want to store the project.
    ```bash
    cd C:\Users\YourName\Documents
    ```

2.  **Clone the Repository:** This downloads the CBAS source code. By default, you will be on the `v2-stable` branch.
    ```bash
    git clone https://github.com/jones-lab-tamu/CBAS.git
    ```

3.  **Navigate into the Project Folder:**
    ```bash
    cd CBAS
    ```

4.  **(IMPORTANT) Choose Which Version to Install and Run**

    The next steps depend entirely on which version you want to use.

    ---
    ### Option A: Install and Run v3 (BETA - Desktop App)

    a. **Switch to the `main` branch:**
        ```bash
        git checkout main
        ```

    b. **Create and Activate a Python Virtual Environment:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

    c. **Install Python Dependencies:** (This will install PyTorch automatically)
        ```bash
        pip install -r requirements.txt
        ```

    d. **Install Node.js Dependencies:**
        ```bash
        npm install
        ```

    e. **Run the Application:**
        ```bash
        npm start
        ```

    ---
    ### Option B: Install and Run v2 (Stable - Web Browser App)

    a. **Make sure you are on the `v2-stable` branch** (this is the default after cloning).

    b. **Create and Activate a Python Virtual Environment:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

    c. **Install General Python Dependencies:**
        ```bash
        pip install -r requirements.txt
        ```
    
    d. **Install PyTorch for CUDA 11.8:**
        ```bash
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```

    e. **Run the Application:**
        ```bash
        python backend/app.py
        ```
    ---

## 5. Updating CBAS

To update your local copy of CBAS to the latest version, follow these steps:

1.  **Open Command Prompt** and navigate to your `CBAS` project folder.
    ```bash
    cd C:\Path\To\Your\CBAS
    ```

2.  **(For v3 users) Activate your virtual environment:**
    ```bash
    .\venv\Scripts\activate
    ```

3.  **Make sure you are on the correct branch.**
    *   For the v3 Desktop App, run: `git checkout main`
    *   For the v2 Web App, run: `git checkout v2-stable`

4.  **Pull the latest code from GitHub:**
    ```bash
    git pull
    ```

5.  **Update your dependencies.** This step is crucial to install any new tools the updated code requires.

    *   **If you are using v3 (the `main` branch):**
        ```bash
        # Update Python packages
        pip install -r requirements.txt
        
        # Update Node.js packages (like Electron)
        npm install
        ```

    *   **If you are using v2 (the `v2-stable` branch):**
        ```bash
        # Update Python packages
        pip install -r requirements.txt
        
        # You may also need to re-run the specific PyTorch install if it has changed
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```