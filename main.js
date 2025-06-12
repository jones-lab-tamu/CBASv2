const { app, BrowserWindow, dialog, ipcMain } = require('electron');
const path = require('path');
const child_process = require('child_process');
const fs = require('fs');

let pythonProcess = null;
let splashWindow = null;
let appWindow = null;

const gotTheLock = app.requestSingleInstanceLock();
if (!gotTheLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    // If someone tries to open a second instance, focus our main app window
    if (appWindow) {
      if (appWindow.isMinimized()) appWindow.restore();
      appWindow.focus();
    }
  });

  function createPythonProcess() {
    const appRoot = app.getAppPath();
    const venvPython = path.join(appRoot, 'venv', 'Scripts', 'python.exe');
    const scriptPath = path.join(appRoot, 'backend', 'app.py');
    const pythonArgs = ['-u', scriptPath];
    console.log(`Spawning Python: "${venvPython}" ${pythonArgs.join(' ')}`);

    if (!fs.existsSync(venvPython)) {
      // Handle error - e.g., show a dialog
      dialog.showErrorBox("Python Error", `Virtual environment not found at ${venvPython}. Please run the installation steps.`);
      app.quit();
      return;
    }
    pythonProcess = child_process.spawn(venvPython, pythonArgs);
    pythonProcess.stdout.on('data', (data) => console.log(`[Python]: ${data.toString().trim()}`));
    pythonProcess.stderr.on('data', (data) => console.error(`[Python ERR]: ${data.toString().trim()}`));
    pythonProcess.on('close', (code) => {
        if (!app.isQuitting) {
            dialog.showErrorBox("Backend Error", `The Python backend process has crashed (code: ${code}). Please restart the application.`);
        }
    });
  }

  function createWindow() {
    // 1. Create the splash screen window FIRST.
    splashWindow = new BrowserWindow({
      width: 500,
      height: 300,
      transparent: true,
      frame: false,
      alwaysOnTop: true,
      webPreferences: {
          nodeIntegration: true,
          contextIsolation: false,
      }
    });
    splashWindow.loadFile(path.join(__dirname, 'frontend/loading.html'));

    // 2. Create the main application window, but keep it hidden.
    appWindow = new BrowserWindow({
      width: 1400,
      height: 900,
      show: false, // <-- Keep it hidden initially
      webPreferences: {
        preload: path.join(__dirname, 'preload.js'),
        contextIsolation: true,
        nodeIntegration: false,
      }
    });

    let webAppUrl = null;
    pythonProcess.stdout.on('data', (data) => {
      const output = data.toString();
      const match = output.match(/Eel server starting on (http:\/\/localhost:\d+)/);
      if (match && !webAppUrl) {
        webAppUrl = match[1];
        console.log(`Eel server detected. Loading URL: ${webAppUrl}`);
        
        // 3. Load the main app into the HIDDEN window.
        appWindow.loadURL(webAppUrl);
      }
    });

    // 4. When the hidden window is fully loaded, show it and close the splash screen.
    appWindow.webContents.on('did-finish-load', () => {
	  // Only perform this logic if the splash screen actually exists.
	  if (splashWindow) {
		appWindow.show();
		splashWindow.close();
		splashWindow = null;
	  }
	});
    
    // IPC for file dialogs remains the same
    ipcMain.on('open-file-dialog', (event) => {
      dialog.showOpenDialog(appWindow, { properties: ['openDirectory'] })
        .then(result => {
          if (!result.canceled) event.sender.send('selected-directory', result.filePaths[0]);
        });
    });
    
    appWindow.on('closed', () => { appWindow = null; });
  }

  // --- Standard Electron lifecycle events ---
  app.on('ready', () => { createPythonProcess(); createWindow(); });
  app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
  app.on('before-quit', () => { app.isQuitting = true; if (pythonProcess) pythonProcess.kill(); });
  app.on('activate', () => { if (appWindow === null && !splashWindow) createWindow(); });
}