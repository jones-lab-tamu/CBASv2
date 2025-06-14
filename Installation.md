# Installing CBAS from source for Windows users

## 1. Install dependencies
   1. Download VLC here: [https://videolan.org](https://videolan.org)
      1. Click Download VLC
      2. Install using default settings
   2. Download Visual Studio here: [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
      1. Click Download Build Tools
      2. Install the Visual Studio Installer using the default settings
      3. Click Desktop development with C++
      4. Click Install
   3. Download Python 3.10 here: [https://python.org/downloads/release/python-3100](https://python.org/downloads/release/python-3100)
      1. Choose Windows installer (64-bit)
      2. Click Add Python 3.10 to PATH
      3. Click Install Now
   4. Download Node.js here: [https://nodejs.org/en](https://nodejs.org/en)
      1. CLick Download Node.js (LTS)
      2. Click Install
      3. Select "Automatically install the necessary tools"
         1. In Command Prompt, do what it tells you ("Press any key...")
         2. A PowerShell window will pop up and install a lot of things, just be patient.
         3. You can close out of it when it stops and says "Authenticode verification returned..."
   5. Download FFMpeg here: [https://gyan.dev/ffmpeg/builds](https://gyan.dev/ffmpeg/builds)
      1. Choose ffmpeg-release-essentials.zip
      2. Unzip the file and move the unzipped folder (ffmpeg-7.0.1-essentials_build) to Documents or another easily accessible location.
      3. Click into the folder and locate the bin folder
      4. Right click on the bin folder and select "Copy as path"
      5. Type "environment variables" into the Windows search bar
      6. Select "Edit the system environment variables"
      7. Select "Environment variables"
      8. Click Path under System Variables
      9. Click Edit...
      10. Click New
      11. Paste the copied FFMpeg path
      12. Click OK
      13. Important: Make sure to click OK and not just X out of the window!
  1.  Download Git here: [https://git-scm.com/download/win](https://git-scm.com/download/win)
         1.  Choose 64-bit git for Windows setup
         2.  Install with the default installation settings

## 2. Install CUDA for GPU optimization (optional but **strongly** recommended, requires NVIDIA GPU)
   1. Download CUDA Toolkit 11.8 here: [https://developer.nvidia.com/cuda-11-8-0-download-archive](https://developer.nvidia.com/cuda-11-8-0-download-archive)
      1. Select Windows, x86_64, 11, exe (local)
      2. In the installation window, select "Express installation"
      3. Click "I understand" when it asks

## 3. Verify installation
   1. Type "command prompt" into the Windows search bar
   2. Start the Command Prompt in Administrator mode (Right Click > Run as Administrator)
   3. Type `python -V`
      1. Note that this is an uppercase V
      2. This should return a version (e.g. `Python 3.10.12`)
   4. Type `node --version`
      1. This should return a version (e.g. `v20.15.1`)
   5. Type `ffmpeg`
      1. This should return text starting with `ffmpeg version...`
   6. If CUDA is installed, type: `nvcc --version`
      1. This should return text starting with `nvcc: NVIDIA (R) Cuda compiler...`

## 4. Install CBAS
   1. Important: in this entire step, be patient and wait until the command is actually finished. For some steps (especially step 10), it may seem like nothing is happening, but it is.
   2. In command prompt, navigate to a directory where you want to house the CBAS code
      1. Example: type `cd C:\Users\Jones-Lab\Documents\`
   3. Type `git clone https://github.com/jones-lab-tamu/CBAS.git cbas-v2`
   4. Enter the CBASv2 folder
      1. Type `cd cbas-v2`
   5. Switch to the stable branch
      1. Type `git checkout v2-stable`
   5. Create a virtual environment for CBAS using Python 3.10
      1. Type `py -3.10 -m venv venv-v2`
   6. Activate virtual environment
      1. Type `.\venv-v2\Scripts\activate`
      2. `(venv-v2)` will be in parentheses in the terminal.
   7. Optional: upgrade pip (not necessary, but the terminal will yell at you if it's out of date)
      1. Type `python -m pip install --upgrade pip`
   8. Install required packages
      1. Type `pip install -r requirements.txt`
      2. Type `npm install`
   9. Install Pytorch
      1.  IF YOU HAVE AN NVIDIA GPU: Type `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
	  2.  IF NOT: Type `pip3 install torch torchvision torchaudio`
  1.  Install transformer models
      1.  Type `pip install -q git+https://github.com/huggingface/transformers.git`
      2.  Remember to be patient!

## 5. Starting CBAS
   1. If finished with Step 4 above
      1. Type `python backend/app.py`
      2. Be patient waiting for it to start for the first time!
   2. If starting from a new command prompt
      1. Enter the CBAS folder
         1. Example: type `cd C:\Users\Jones-Lab\Documents\CBAS`
      2. Type `dir`
      3. Type `.\venv-v2\Scripts\activate`
      4. Type `python backend/app.py`
   
## 6. Automatically updating CBAS (when needed)
   1. In command prompt, enter the CBASv2 folder
      1. Example: type `cd C:\Users\Jones-Lab\Documents\CBAS`
   2. Type `git pull origin v2-stable`


