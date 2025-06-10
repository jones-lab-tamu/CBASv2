const { app, BrowserWindow, dialog, ipcMain } = require('electron');
const path = require('path');
const child_process = require('child_process');
const fs = require('fs');

let pythonProcess = null;
let mainWindow = null;

const gotTheLock = app.requestSingleInstanceLock();
if (!gotTheLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });

  function createPythonProcess() {
    const appRoot = app.getAppPath();
    const venvPython = path.join(appRoot, 'venv', 'Scripts', 'python.exe');
    const scriptPath = path.join(appRoot, 'backend', 'app.py');
    const pythonArgs = ['-u', scriptPath];
    console.log(`Spawning Python: "${venvPython}" ${pythonArgs.join(' ')}`);
    if (!fs.existsSync(venvPython)) { /* error handling */ }
    pythonProcess = child_process.spawn(venvPython, pythonArgs);
    pythonProcess.stdout.on('data', (data) => console.log(`[Python]: ${data.toString().trim()}`));
    pythonProcess.stderr.on('data', (data) => console.error(`[Python ERR]: ${data.toString().trim()}`));
    pythonProcess.on('close', (code) => { /* error handling */ });
  }

  function createWindow() {
    mainWindow = new BrowserWindow({
      width: 1400,
      height: 900,
      webPreferences: {
        // Point to the preload script.
        preload: path.join(app.getAppPath(), 'preload.js'),
        // These are best practices for security.
        contextIsolation: true, // Keep this true.
        nodeIntegration: false, // Keep this false.
      }
    });

    // --- We go back to loadURL, but will fix the CSS pathing ---
    // The preload script makes this safe.
    // We will find the port from the Python output first.
    let webAppUrl = null;
    pythonProcess.stdout.on('data', (data) => {
      const output = data.toString();
      const match = output.match(/Eel server starting on (http:\/\/localhost:\d+)/);
      if (match && !webAppUrl) {
        webAppUrl = match[1];
        console.log(`Eel server detected. Loading URL: ${webAppUrl}`);
        mainWindow.loadURL(webAppUrl);
      }
    });
    
    ipcMain.on('open-file-dialog', (event) => {
      dialog.showOpenDialog({ properties: ['openDirectory'] })
        .then(result => {
          if (!result.canceled) event.sender.send('selected-directory', result.filePaths[0]);
        });
    });
    
    mainWindow.on('closed', () => { mainWindow = null; });
  }

  // --- Standard Electron lifecycle events ---
  app.on('ready', () => { createPythonProcess(); createWindow(); });
  app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
  app.on('before-quit', () => { app.isQuitting = true; if (pythonProcess) pythonProcess.kill(); });
  app.on('activate', () => { if (mainWindow === null) createWindow(); });
}