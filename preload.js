const { contextBridge, ipcRenderer } = require('electron');

// Expose a safe, limited version of ipcRenderer to the renderer process (your HTML pages).
// The frontend code can now use `window.electronAPI.send(...)` and `window.electronAPI.on(...)`.
contextBridge.exposeInMainWorld('electronAPI', {
  send: (channel, data) => {
    // Whitelist channels to prevent exposing all of IPC
    let validChannels = ['open-file-dialog'];
    if (validChannels.includes(channel)) {
      ipcRenderer.send(channel, data);
    }
  },
  on: (channel, func) => {
    let validChannels = ['selected-directory'];
    if (validChannels.includes(channel)) {
      // Deliberately strip event as it includes `sender`
      ipcRenderer.on(channel, (event, ...args) => func(...args));
    }
  }
});