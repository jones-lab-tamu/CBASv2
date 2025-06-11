// main.js

// Modules to control application life and create native browser window
const { app, BrowserWindow } = require('electron')
const path = require('node:path')

const port = process.argv[2] || 8000

const createWindow = () => {
    // Create the browser window.
    const mainWindow = new BrowserWindow({
    width: 800,
    height: 1000,
    webPreferences: {
        preload: path.join(__dirname, 'preload.js'),
        nodeIntegration: true,
        contextIsolation: false
    },
    autoHideMenuBar: true
    })

    // and load the index.html of the app.
    mainWindow.loadURL(`http://localhost:${port}/index.html`)

    // Open the DevTools.
    // mainWindow.webContents.openDevTools()
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {

  createWindow()

  app.on('activate', () => {
    // On macOS it's common to re-create a window in the app when the
    // dock icon is clicked and there are no other windows open.
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })


})

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})


const ipc = require('electron').ipcMain
const dialog = require('electron').dialog
const storage = require('electron-json-storage');
const { electron } = require('node:process')

ipc.on('open-file-dialog', async function (event) {

    let directory = await dialog.showOpenDialog({
        properties: ['openDirectory']
    })

    if (directory['canceled']) {
        event.sender.send('selected-directory', null)
    } else {
        event.sender.send('selected-directory', directory['filePaths'][0])
    }
})

ipc.on('select-videos-dialog', async function (event) { 
    let result = await dialog.showOpenDialog({
        properties: ['openFile', 'multiSelections'],
        filters: [
            { name: 'Videos', extensions: ['mp4'] }
        ]
    });

    if (!result.canceled) {
        event.sender.send('selected-videos', result.filePaths);
    }
});