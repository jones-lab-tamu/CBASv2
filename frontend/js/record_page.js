/**
 * @file Manages the Record page UI and interactions.
 * @description This file handles camera listing, settings, starting/stopping recordings,
 * and displaying camera thumbnails and status.
 */

// =================================================================
// EEL-EXPOSED & LOG PANEL FUNCTIONS
// =================================================================

eel.expose(update_log_panel);
function update_log_panel(message) {
    const logContainer = document.getElementById('log-panel-content');
    if (!logContainer) return;

    // --- Logic for sessionStorage ---
    let logHistory = JSON.parse(sessionStorage.getItem('logHistory') || '[]');
    logHistory.push(message);
    while (logHistory.length > 500) {
        logHistory.shift();
    }
    sessionStorage.setItem('logHistory', JSON.stringify(logHistory));
    // --- End of sessionStorage logic ---

    renderLogMessage(message, logContainer);
    logContainer.scrollTop = logContainer.scrollHeight;
}

/**
 * Helper function to render a single log message to the DOM.
 * @param {string} message - The log message text.
 * @param {HTMLElement} container - The container to append the message to.
 */
function renderLogMessage(message, container) {
    const logEntry = document.createElement('div');
    logEntry.className = 'log-message';

    if (message.includes('[ERROR]')) {
        logEntry.classList.add('log-level-ERROR');
    } else if (message.includes('[WARN]')) {
        logEntry.classList.add('log-level-WARN');
    } else {
        logEntry.classList.add('log-level-INFO');
    }
    
    logEntry.textContent = message;
    container.appendChild(logEntry);
}


document.addEventListener('DOMContentLoaded', () => {
    const logContainer = document.getElementById('log-panel-content');
    if (logContainer) {
        const logHistory = JSON.parse(sessionStorage.getItem('logHistory') || '[]');
        logHistory.forEach(msg => renderLogMessage(msg, logContainer));
        logContainer.scrollTop = logContainer.scrollHeight;
    }
    
    const clearBtn = document.getElementById('clear-log-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            if (logContainer) {
                logContainer.innerHTML = '';
                sessionStorage.setItem('logHistory', '[]'); 
                update_log_panel('Log cleared.');
            }
        });
    }
});

// =================================================================
// GLOBAL STATE & VARIABLES
// =================================================================

let routingInProgress = false;
let originalCameraNameForSettings = "";
let modalPreviewImage = new Image();

const addCameraModalElement = document.getElementById('addCamera');
const statusModalElement = document.getElementById('statusModal');
const cameraSettingsModalElement = document.getElementById('cameraSettings');
const errorModalElement = document.getElementById('errorModal');

let addCameraBsModal = addCameraModalElement ? new bootstrap.Modal(addCameraModalElement) : null;
let statusBsModal = statusModalElement ? new bootstrap.Modal(statusModalElement) : null;
let cameraSettingsBsModal = cameraSettingsModalElement ? new bootstrap.Modal(cameraSettingsModalElement) : null;
let generalErrorBsModal = errorModalElement ? new bootstrap.Modal(errorModalElement) : null;


// =================================================================
// ROUTING & UTILITY FUNCTIONS
// =================================================================

function routeToRecordPage() { routingInProgress = true; window.location.href = './record.html'; }
function routeToLabelTrainPage() { routingInProgress = true; window.location.href = './label-train.html'; }
function routeToVisualizePage() { routingInProgress = true; window.location.href = './visualize.html'; }

function showErrorOnRecordPage(message) {
    const errorMessageElement = document.getElementById("error-message");
    if (errorMessageElement && generalErrorBsModal) {
        errorMessageElement.innerText = message;
        generalErrorBsModal.show();
    } else {
        alert(message);
    }
}


// =================================================================
// EEL-EXPOSED FUNCTIONS (Called FROM Python)
// =================================================================

eel.expose(updateImageSrc);
async function updateImageSrc(cameraName, base64Val) {
    const canvas = document.getElementById(`camera-${cameraName}`);
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    if (!base64Val) {
        const placeholderImg = new Image();
        placeholderImg.src = "assets/noConnection.png";
        placeholderImg.onload = () => { ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.drawImage(placeholderImg, 0, 0, canvas.width, canvas.height); };
        return;
    }

    const img = new Image();
    img.src = "data:image/jpeg;base64," + base64Val;
    canvas.setAttribute("cbas_image_source", img.src);

    img.onload = async () => {
        try {
            const settings = await eel.get_camera_settings(cameraName)();
            if (!settings) return;
            const s = {
                x: Math.max(0, Math.min(1, parseFloat(settings.crop_left_x) || 0)),
                y: Math.max(0, Math.min(1, parseFloat(settings.crop_top_y) || 0)),
                w: Math.max(0, Math.min(1, parseFloat(settings.crop_width) || 1)),
                h: Math.max(0, Math.min(1, parseFloat(settings.crop_height) || 1))
            };
            const cropX = s.x * img.naturalWidth;
            const cropY = s.y * img.naturalHeight;
            const cropW = s.w * img.naturalWidth;
            const cropH = s.h * img.naturalHeight;
            const res = parseInt(settings.resolution) || 256;
            drawImageOnCanvas(img, ctx, cropX, cropY, cropW, cropH, res);
        } catch (error) {
            console.error(`Error in updateImageSrc for ${cameraName}:`, error);
        }
    };
    img.onerror = () => console.error(`Failed to load image data for camera ${cameraName}.`);
}


// =================================================================
// UI INTERACTION & EVENT HANDLERS (Called FROM HTML)
// =================================================================

function showAddCameraModal() {
    addCameraBsModal?.show();
}

async function showStatusModal() {
    try {
        const status = await eel.get_cbas_status()();
        document.getElementById("status-streams").innerText = status.streams ? "Recording: " + status.streams.join(", ") : "No cameras recording.";
        document.getElementById("status-encode-count").innerText = "Files to encode: " + status.encode_file_count;
        statusBsModal?.show();
    } catch (e) { console.error("Get status error:", e); }
}

async function addCameraSubmit() {
    const name = document.getElementById('camera-name-modal-input').value;
    const rtsp = document.getElementById('rtsp-url-modal-input').value;
    if (!name.trim() || !rtsp.trim()) { showErrorOnRecordPage('Name and RTSP URL are required.'); return; }
    
    const success = await eel.create_camera(name, rtsp)();
    if (success) {
        addCameraBsModal?.hide();
        document.getElementById('camera-name-modal-input').value = "";
        document.getElementById('rtsp-url-modal-input').value = "";
        await loadCameras();
    } else {
        showErrorOnRecordPage(`Failed to create camera '${name}'. It may already exist.`);
    }
}

async function startCamera(cameraName) {
    const sessionName = document.getElementById('session-name-input').value;
    if (!sessionName.trim()) {
        showErrorOnRecordPage('Please enter a Session Name before starting a recording.');
        return;
    }
    const success = await eel.start_camera_stream(cameraName, sessionName, 600)();
    if (success) {
        document.getElementById(`before-recording-${cameraName}`).style.display = 'none';
        document.getElementById(`during-recording-${cameraName}`).style.display = 'flex';
    } else {
        showErrorOnRecordPage(`'${cameraName}' might already be recording or an error occurred.`);
    }
    await updateCamButtons();
}

async function stopCamera(cameraName) {
    const success = await eel.stop_camera_stream(cameraName)();
    if (success) {
        document.getElementById(`during-recording-${cameraName}`).style.display = 'none';
        document.getElementById(`before-recording-${cameraName}`).style.display = 'flex';
    } else {
        showErrorOnRecordPage(`Could not stop ${cameraName}. Is it recording?`);
    }
    await updateCamButtons();
}

async function startAllCameras() {
    const cameras = await eel.get_camera_list()();
    if (cameras?.length > 0) {
        for (const [name] of cameras) await startCamera(name);
        await updateCamButtons();
    } else {
        alert("No cameras are configured to start.");
    }
}

async function stopAllCameras() {
    const activeStreams = await eel.get_active_streams()() || [];
    if (activeStreams.length > 0) {
        for (const name of activeStreams) await stopCamera(name);
        await updateCamButtons();
    }
}

function liveViewCamera(cameraName) {
    eel.open_camera_live_view(cameraName)();
}

async function saveCameraSettings() {
    const newName = document.getElementById('cs-name').value;
    if (!newName.trim()) { showErrorOnRecordPage("Camera name cannot be empty."); return; }
    
    const settings = {
        "rtsp_url": document.getElementById('cs-url').value,
        "framerate": parseInt(document.getElementById('cs-framerate').value) || 10,
        "resolution": parseInt(document.getElementById('cs-resolution').value) || 256,
        'crop_left_x': parseFloat(document.getElementById('cs-cropx').value) || 0,
        'crop_top_y': parseFloat(document.getElementById('cs-cropy').value) || 0,
        'crop_width': parseFloat(document.getElementById('cs-crop-width').value) || 1,
        'crop_height': parseFloat(document.getElementById('cs-crop-height').value) || 1,
    };

    let cameraToUpdate = originalCameraNameForSettings;
    if (newName !== originalCameraNameForSettings) {
        const renameSuccess = await eel.rename_camera(originalCameraNameForSettings, newName)();
        if (!renameSuccess) { showErrorOnRecordPage(`Failed to rename. '${newName}' may exist.`); return; }
        cameraToUpdate = newName;
    }
    
    const saveSuccess = await eel.save_camera_settings(cameraToUpdate, settings)();
    if (saveSuccess) {
        cameraSettingsBsModal?.hide();
        await loadCameras();
    } else {
        showErrorOnRecordPage("Failed to save camera settings.");
    }
}


// =================================================================
// CORE APPLICATION LOGIC
// =================================================================

async function loadCameraHTMLCards() {
    const container = document.getElementById('camera-container');
    if (!container) return;
    container.innerHTML = "";

    const cameras = await eel.get_camera_list()();
    if (!cameras || cameras.length === 0) {
        container.innerHTML = "<div class='col'><p class='text-light text-center mt-3'>No cameras configured. Click the '+' button to add one.</p></div>";
        return;
    }

    for (const [name, settings] of cameras) {
        const isCropped = settings.crop_left_x !== 0 || settings.crop_top_y !== 0 || settings.crop_width !== 1 || settings.crop_height !== 1;
        const displayName = isCropped ? name : `${name} <small class='text-muted'>(uncropped)</small>`;
        
        container.innerHTML += `
            <div class="col-auto mb-3">
                <div class="card shadow text-white bg-dark" style="width: 320px;">
                    <div class="card-header py-2"><h5 class="card-title mb-0">${displayName}</h5></div>
                    <canvas id="camera-${name}" width="300" height="225" style="background-color: #343a40; display: block; margin: auto; margin-top:10px;"></canvas>
                    <div class="card-footer d-flex justify-content-center p-2">
                        <div id="before-recording-${name}" style="display: flex;">
                            <button class="btn btn-sm btn-outline-light me-1" onclick="loadCameraSettings('${name}')" title="Settings/Crop"><i class="bi bi-crop"></i></button>
                            <button class="btn btn-sm btn-outline-light me-1" onclick="liveViewCamera('${name}')" title="Live View (VLC)"><i class="bi bi-eye-fill"></i></button>
                            <button class="btn btn-sm btn-success" onclick="startCamera('${name}')" title="Start Recording"><i class="bi bi-camera-video-fill"></i> Start</button>
                        </div>
                        <div id="during-recording-${name}" style="display: none;">
                            <button class="btn btn-sm btn-outline-light me-1" onclick="liveViewCamera('${name}')" title="Live View (VLC)"><i class="bi bi-eye-fill"></i></button>
                            <button class="btn btn-sm btn-danger" onclick="stopCamera('${name}')" title="Stop Recording"><i class="bi bi-square-fill"></i> Stop</button>
                        </div>
                    </div>
                </div>
            </div>`;
    }
    cameras.forEach(([name]) => {
        const canvas = document.getElementById(`camera-${name}`);
        if(canvas) {
            const ctx = canvas.getContext("2d");
            const img = new Image();
            img.src = "assets/noConnection.png";
            img.onload = () => ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        }
    });
}

async function loadCameras() {
    await loadCameraHTMLCards();
    await updateCamButtons();
    eel.download_camera_thumbnails()();
}

async function updateCamButtons() {
    const activeStreams = await eel.get_active_streams()() || [];
    setRecordAllIcon(activeStreams.length > 0);

    const cameras = await eel.get_camera_list()();
    if (cameras) {
        cameras.forEach(([name]) => {
            const beforeRec = document.getElementById(`before-recording-${name}`);
            const duringRec = document.getElementById(`during-recording-${name}`);
            if (beforeRec && duringRec) {
                const isActive = activeStreams.includes(name);
                beforeRec.style.display = isActive ? 'none' : 'flex';
                duringRec.style.display = isActive ? 'flex' : 'none';
            }
        });
    }
}

async function updateStatusIcon() {
    const icon = document.getElementById("status-camera-icon");
    if (!icon) return;
    const status = await eel.get_cbas_status()();
    const isStreaming = status && status.streams && status.streams.length > 0;
    icon.style.color = isStreaming ? "red" : "white";
    isStreaming ? icon.classList.add("blinking") : icon.classList.remove("blinking");
}

function setRecordAllIcon(isAnyRecording) {
    const fabContainer = document.querySelector('.fab-container-right');
    if (!fabContainer) return;
    const fabButton = Array.from(fabContainer.children).find(child => child.classList.contains('fab') && child.getAttribute('onclick')?.includes('All'));
    if (!fabButton) return;
    const fabIcon = fabButton.querySelector('i');
    if (!fabIcon) return;

    if (isAnyRecording) {
        fabIcon.className = 'bi bi-square-fill';
        fabButton.onclick = stopAllCameras;
        fabButton.title = "Stop All Recordings";
    } else {
        fabIcon.className = 'bi bi-camera-video-fill';
        fabButton.onclick = startAllCameras;
        fabButton.title = "Start All Recordings";
    }
}

function drawImageOnCanvas(img, ctx, sx, sy, sw, sh, targetResolution) {
    const canvas = ctx.canvas;
    if (sw <= 0 || sh <= 0) { ctx.clearRect(0, 0, canvas.width, canvas.height); return; }
    const ratio = Math.min(canvas.width / sw, canvas.height / sh);
    const drawW = sw * ratio, drawH = sh * ratio;
    const destX = (canvas.width - drawW) / 2;
    const destY = (canvas.height - drawH) / 2;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, sx, sy, sw, sh, destX, destY, drawW, drawH);
}

async function loadCameraSettings(cameraName) {
    const modalCanvas = document.getElementById("camera-image");
    if (!modalCanvas) return;
    const modalCtx = modalCanvas.getContext("2d");

    try {
        const settings = await eel.get_camera_settings(cameraName)();
        if (settings) {
            originalCameraNameForSettings = cameraName;
            document.getElementById('cs-name').value = cameraName;
            document.getElementById('cs-url').value = settings.rtsp_url;
            document.getElementById('cs-framerate').value = settings.framerate;
            document.getElementById('cs-resolution').value = settings.resolution;
            document.getElementById('cs-cropx').value = settings.crop_left_x;
            document.getElementById('cs-cropy').value = settings.crop_top_y;
            document.getElementById('cs-crop-width').value = settings.crop_width;
            document.getElementById('cs-crop-height').value = settings.crop_height;

            const mainCanvasImageSrc = document.getElementById(`camera-${cameraName}`)?.getAttribute("cbas_image_source");
            
            modalPreviewImage.onload = () => {
                const aspectRatio = modalPreviewImage.naturalWidth / modalPreviewImage.naturalHeight;
                modalCanvas.width = 600;
                modalCanvas.height = modalCanvas.width / aspectRatio;
                drawBoundsOnModalCanvas(modalPreviewImage);
            };
            modalPreviewImage.src = mainCanvasImageSrc || "assets/noConnection.png";

            cameraSettingsBsModal?.show();
        } else {
            showErrorOnRecordPage(`Could not load settings for camera: ${cameraName}`);
        }
    } catch (error) {
        showErrorOnRecordPage(`Failed to load settings: ${error.message || error}`);
    }
}

function drawBoundsOnModalCanvas(imageToDraw) {
    const modalCanvas = document.getElementById("camera-image");
    if (!modalCanvas || !imageToDraw || !imageToDraw.complete || imageToDraw.naturalWidth === 0) return;
    const modalCtx = modalCanvas.getContext("2d");

    let cx = parseFloat(document.getElementById('cs-cropx').value) || 0;
    let cy = parseFloat(document.getElementById('cs-cropy').value) || 0;
    let cw = parseFloat(document.getElementById('cs-crop-width').value) || 1;
    let ch = parseFloat(document.getElementById('cs-crop-height').value) || 1;

    cx = Math.max(0, Math.min(1, cx));
    cy = Math.max(0, Math.min(1, cy));
    cw = Math.min(cw, 1 - cx);
    ch = Math.min(ch, 1 - cy);

    const sx = cx * imageToDraw.naturalWidth, sy = cy * imageToDraw.naturalHeight;
    const sw = cw * imageToDraw.naturalWidth, sh = ch * imageToDraw.naturalHeight;

    modalCtx.clearRect(0, 0, modalCanvas.width, modalCanvas.height);
    modalCtx.drawImage(imageToDraw, 0, 0, modalCanvas.width, modalCanvas.height);
    
    if (sw > 0 && sh > 0) {
        modalCtx.strokeStyle = 'rgba(255, 0, 0, 0.9)';
        modalCtx.lineWidth = 2;
        modalCtx.strokeRect(
            (sx / imageToDraw.naturalWidth) * modalCanvas.width, 
            (sy / imageToDraw.naturalHeight) * modalCanvas.height, 
            (sw / imageToDraw.naturalWidth) * modalCanvas.width, 
            (sh / imageToDraw.naturalHeight) * modalCanvas.height
        );
    }
}


// =================================================================
// PAGE INITIALIZATION
// =================================================================

document.addEventListener('DOMContentLoaded', () => {
    loadCameras();
    setInterval(updateStatusIcon, 3000);
    document.getElementById('addCameraButton')?.addEventListener('click', addCameraSubmit);
    
    const settingIds = ['cs-cropx', 'cs-cropy', 'cs-crop-width', 'cs-crop-height'];
    settingIds.forEach(id => {
        document.getElementById(id)?.addEventListener('input', () => drawBoundsOnModalCanvas(modalPreviewImage));
    });

    const logCollapseElement = document.getElementById('log-panel-collapse');
    if (logCollapseElement) {
        const fabLeft = document.querySelector('.fab-container-left');
        const fabRight = document.querySelector('.fab-container-right');
        const fabUpPosition = `${200 + 45 + 5}px`; 
        const fabDownPosition = '65px';

        logCollapseElement.addEventListener('show.bs.collapse', () => {
            if (fabLeft) fabLeft.style.bottom = fabUpPosition;
            if (fabRight) fabRight.style.bottom = fabUpPosition;
        });
        logCollapseElement.addEventListener('hide.bs.collapse', () => {
            if (fabLeft) fabLeft.style.bottom = fabDownPosition;
            if (fabRight) fabRight.style.bottom = fabDownPosition;
        });
    }
});

window.addEventListener("unload", () => {
    if (!routingInProgress) { eel.kill_streams()?.catch(err => console.error(err)); }
});
window.onbeforeunload = () => {
    if (!routingInProgress) { eel.kill_streams()?.catch(err => console.error(err)); }
};