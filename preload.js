const { contextBridge, ipcRenderer } = require('electron');

// Expose a safe, limited version of ipcRenderer to the renderer process (your HTML pages).
// The frontend code can now use `window.electronAPI.send(...)` and `window.electronAPI.on(...)`.
contextBridge.exposeInMainWorld('electronAPI', {
  send: (channel, data, data2) => { // Modify to accept more args
    let validChannels = ['open-file-dialog', 'save-file-to-disk']; // Add the new channel
    if (validChannels.includes(channel)) {
      ipcRenderer.send(channel, data, data2); // Pass both args
    }
  },
  invoke: (channel, data) => {
    // Replace 'show-save-dialog' with our new, specific channel name
    let validChannels = ['show-folder-dialog']; 
    if (validChannels.includes(channel)) {
      return ipcRenderer.invoke(channel, data);
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