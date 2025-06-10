# Installing CBAS v3 (BETA) from Source (Windows)

This guide provides step-by-step instructions for installing the CBAS application on a Windows machine with an NVIDIA GPU.

## 1. Install Required Software

You will need to install several pieces of software before setting up CBAS.

1.  **Git:** The version control system used to download the code.
    *   Download from: [https://git-scm.com/download/win](https://git-scm.com/download/win)
    *   Run the installer and use all the default settings.

2.  **Python:** The core programming language for the backend. We recommend Python 3.10.
    *   Download from: [https://www.python.org/downloads/release/python-31011/](https://www.python.org/downloads/release/python-31011/) (scroll down and select "Windows installer (64-bit)").
    *   During installation, **make sure to check the box that says "Add Python 3.10 to PATH"**.
    *   Choose "Install Now".

3.  **FFmpeg:** A powerful tool for video processing that CBAS uses for recording and thumbnail generation.
    *   Download the "essentials" build from: [https://gyan.dev/ffmpeg/builds/](https://gyan.dev/ffmpeg/builds/) (e.g., `ffmpeg-release-essentials.zip`).
    *   Unzip the downloaded file.
    *   Move the resulting folder (e.g., `ffmpeg-7.0-essentials_build`) to a permanent location, like `C:\`.
    *   Add the `bin` folder inside it to your Windows PATH environment variable.
        *   Search for "Environment Variables" in the Windows search bar and open it.
        *   Click "Environment Variables...".
        *   Under "System variables", find and select the `Path` variable, then click "Edit...".
        *   Click "New" and paste the full path to the `bin` folder (e.g., `C:\ffmpeg-7.0-essentials_build\bin`).
        *   Click OK on all windows to save.

4.  **VLC Media Player (Optional):** Used for the "Live View" feature.
    *   Download from: [https://www.videolan.org/vlc/](https://www.videolan.org/vlc/)
    *   Install using the default settings.

## 2. Install NVIDIA CUDA Toolkit (Strongly Recommended for GPU)

To use your NVIDIA GPU for fast model training and inference, you need the CUDA Toolkit. CBAS is optimized for **CUDA 11.8**.

1.  **Download CUDA Toolkit 11.8:** [https://developer.nvidia.com/cuda-11-8-0-download-archive](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2.  Select the options: *Windows -> x86_64 -> 10 or 11 -> exe (local)*.
3.  Download and run the installer. Choose the **"Express (Recommended)"** installation option.

## 3. Verify Your Installations

Open a **new** Command Prompt (`cmd`) and run the following commands to ensure everything is installed correctly.

*   `git --version` (Should show a Git version)
*   `python --version` (Should show Python 3.10.x)
*   `ffmpeg -version` (Should show ffmpeg version info)
*   `nvcc --version` (If you installed CUDA, this should show the CUDA version 11.8)

## 4. Install the CBAS Application

Now, we will download the CBAS source code and set up its Python environment.

1.  **Open Command Prompt** and navigate to the directory where you want to store the project (e.g., your Documents folder).
    ```bash
    cd C:\Users\YourName\Documents
    ```

2.  **Clone the Repository:** This downloads the CBAS source code.
    ```bash
    git clone https://github.com/jones-lab-tamu/CBAS.git
    ```

3.  **Navigate into the Project Folder:**
    ```bash
    cd CBAS
    ```

4.  **(IMPORTANT) Choose Your Branch:**
    *   For the latest **stable v2**, stay on the default branch: `v2-stable`.
    *   To use the **new v3 beta**, switch to the main branch:
        ```bash
        git checkout main
        ```

5.  **Create and Activate a Python Virtual Environment:** This creates an isolated space for CBAS's Python packages.
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
    You will see `(venv)` appear at the beginning of your command prompt line.

6.  **Install All Required Python Packages:** This single command reads the `requirements.txt` file and installs everything needed, including PyTorch. **This may take several minutes.**
    ```bash
    pip install -r requirements.txt
    ```

## 5. Starting CBAS

Once the installation is complete, you can start the application.

1.  Make sure you are in the `CBAS` directory and your virtual environment is active (`(venv)` is visible).
2.  Run the main application script:
    ```bash
    python backend/app.py
    ```
3.  Be patient on the first launch as it sets things up. The CBAS GUI window should appear.

## 6. Updating CBAS

To get the latest updates for the branch you are on, navigate to the `CBAS` directory in your command prompt and run:

```bash
git pull