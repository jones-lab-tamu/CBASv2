# Installing CBAS from Source (Windows)

This guide provides step-by-step instructions for installing the CBAS application on a Windows machine. The primary focus is on the modern v3 (Beta) desktop application.

## Section 1: Installing CBAS v3 (BETA - Recommended)

These are the complete instructions for installing the current-generation desktop application.

### Step 1.1: Install Primary Dependencies

1.  **Git:** The version control system used to download the code.
    *   Download from: [https://git-scm.com/download/win](https://git-scm.com/download/win) and install using the default settings.

2.  **Python (64-bit, version 3.11):** This specific version is required for compatibility with PyTorch.
    *   Download the **"Windows installer (64-bit)"** for Python 3.11 from here: [https://www.python.org/downloads/release/python-3119/](https://www.python.org/downloads/release/python-3119/)
    *   During installation, **it is essential that you check the box that says "Add python.exe to PATH"**.

3.  **Node.js (LTS version):** Required for the Electron user interface.
    *   Download the LTS version from: [https://nodejs.org/](https://nodejs.org/). Run the installer with default settings.

4.  **FFmpeg:** A powerful tool for video processing used by CBAS for recording.
    *   Download the "essentials" build from: [https://gyan.dev/ffmpeg/builds/](https://gyan.dev/ffmpeg/builds/).
    *   Unzip the file and move the resulting folder to a permanent location (e.g., `C:\`).
    *   Add its `bin` folder to your Windows PATH environment variable (e.g., `C:\ffmpeg-7.0-essentials_build\bin`).

### Step 1.2: Install and Run CBAS v3

1.  **Open Command Prompt** and navigate to where you want to store the project.
    ```
    cd C:\Users\YourName\Documents
    ```

2.  **Clone the Repository and Checkout the `main` Branch:**
    ```
    git clone https://github.com/jones-lab-tamu/CBAS.git
    cd CBAS
    git checkout main
    ```

3.  **Create and Activate a Python Virtual Environment:**
    ```
    python -m venv venv
    .\venv\Scripts\activate
    ```

4.  **Install All Python Dependencies (including PyTorch for GPU):**
    This single step will install all necessary packages. The PyTorch command is for modern NVIDIA GPUs (RTX 20-series and newer).
    ```
    pip install -r requirements.txt
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

5.  **Install Node.js Dependencies:**
    ```
    npm install
    ```

6.  **Run the Application:**
    ```
    npm start
    ```

---

## Section 2: Installing Legacy CBAS v2 (For Advanced Users)

**Note:** The legacy v2 version is a browser-based application with different dependencies. It is recommended to install v2 in a completely separate folder and virtual environment to avoid conflicts with the modern v3 application. These instructions require installing the older CUDA Toolkit 11.8.

<details>
<summary>Click to expand v2 Installation Instructions</summary>

1.  **Install NVIDIA CUDA Toolkit 11.8:** Download and install from [NVIDIA's archive](https://developer.nvidia.com/cuda-11-8-0-download-archive). Use the "Express" installation.
2.  **Install Visual Studio Build Tools:** Download from [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select the "Desktop development with C++" workload.
3.  **Clone and Checkout `v2-stable` Branch:**
    ```
    git clone https://github.com/jones-lab-tamu/CBAS.git cbas-v2
    cd cbas-v2
    git checkout v2-stable
    ```
4.  **Create and Activate a Separate Virtual Environment:**
    ```
    python -m venv venv-v2
    .\venv-v2\Scripts\activate
    ```
5.  **Install Python Dependencies for v2:**
    ```
    pip install -r requirements.txt
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
6.  **Run the Application:**
    ```
    python backend/app.py
    ```

</details>