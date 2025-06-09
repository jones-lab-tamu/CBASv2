/**
 * @file record_page.js
 * Manages the recording page UI and interactions.
 * Handles camera listing, settings, starting/stopping recordings,
 * and displaying camera thumbnails and status.
 */

// const ipc = window.ipcRenderer; // Not directly used in this file, but good to keep if future IPC is needed

/** Flag to prevent `kill_streams` from being called during intentional page navigation. */
let routingInProgress = false;

// Initialize Bootstrap Modals safely
const addCameraModalElement = document.getElementById('addCamera');
let addCameraBsModal = addCameraModalElement ? new bootstrap.Modal(addCameraModalElement) : null;

const statusModalElement = document.getElementById('statusModal');
let statusBsModal = statusModalElement ? new bootstrap.Modal(statusModalElement) : null;

const cameraSettingsModalElement = document.getElementById('cameraSettings');
let cameraSettingsBsModal = cameraSettingsModalElement ? new bootstrap.Modal(cameraSettingsModalElement) : null;

const errorModalElement = document.getElementById('errorModal'); // General error modal for this page
const errorMessageElement = document.getElementById("error-message");
let generalErrorBsModal = errorModalElement ? new bootstrap.Modal(errorModalElement) : null;

/** Stores the original camera name when opening settings, for rename detection. */
let originalCameraNameForSettings = "";
/** Stores the current image for the settings modal preview */
let modalPreviewImage = new Image();


/** Shows the error modal with a custom message. */
function showErrorOnRecordPage(message) {
    if (errorMessageElement && generalErrorBsModal) {
        errorMessageElement.innerText = message;
        generalErrorBsModal.show();
    } else {
        alert(message); // Fallback
    }
}

/** Navigates to the record page (self). */
function routeToRecordPage() { routingInProgress = true; window.location.href = './record.html'; }
/** Navigates to the label & train page. */
function routeToLabelTrainPage() { routingInProgress = true; window.location.href = './label-train.html'; }
/** Navigates to the visualize page. */
function routeToVisualizePage() { routingInProgress = true; window.location.href = './visualize.html'; }

/**
 * Dynamically loads camera cards into the UI based on data from the backend.
 */
async function loadCameraHTMLCards() {
    const container = document.getElementById('camera-container');
    if (!container) { console.error("Camera container element not found!"); return; }
    container.innerHTML = ""; // Clear existing camera cards

    try {
        const cameras = await eel.get_camera_list()();
        if (!cameras || cameras.length === 0) {
            container.innerHTML = "<div class='col'><p class='text-light text-center mt-3'>No cameras configured. Click the '+' button below to add a new camera.</p></div>";
            return;
        }

        for (const [cameraName, cameraSettings] of cameras) {
            let displayName = cameraName;
            const cx = parseFloat(cameraSettings['crop_left_x']);
            const cy = parseFloat(cameraSettings['crop_top_y']);
            const cw = parseFloat(cameraSettings['crop_width']);
            const ch = parseFloat(cameraSettings['crop_height']);
            const isCropped = cx !== 0 || cy !== 0 || cw !== 1 || ch !== 1;
            if (!isCropped) {
                displayName += " <small class='text-muted'>(uncropped)</small>";
            }

            container.innerHTML += `
                <div class="col-auto mb-3">
                    <div class="card shadow text-white bg-dark" style="width: 320px;"> <!-- Fixed width for consistency -->
                        <div class="card-header py-2"> <!-- Reduced padding -->
                            <h5 class="card-title mb-0">${displayName}</h5>
                        </div>
                        <canvas id="camera-${cameraName}" width="300" height="225" style="background-color: #343a40; display: block; margin: auto; margin-top:10px;"></canvas> <!-- Adjusted canvas size and centering -->
                        <div class="card-footer d-flex justify-content-center p-2">
                            <div id="before-recording-${cameraName}" style="display: flex;"> <!-- Use display:flex for button group -->
                                <button class="btn btn-sm btn-outline-light me-1" onclick="loadCameraSettings('${cameraName}')" title="Crop/Settings"><i class="bi bi-crop"></i></button>
                                <button class="btn btn-sm btn-outline-light me-1" onclick="liveViewCamera('${cameraName}')" title="Live View (VLC)"><i class="bi bi-eye-fill"></i></button>
                                <button class="btn btn-sm btn-success" onclick="startCamera('${cameraName}')" title="Start Recording"><i class="bi bi-camera-video-fill"></i> Start</button>
                            </div>
                            <div id="during-recording-${cameraName}" style="display: none;"> <!-- Use display:none for button group -->
                                <button class="btn btn-sm btn-outline-light me-1" onclick="liveViewCamera('${cameraName}')" title="Live View (VLC)"><i class="bi bi-eye-fill"></i></button>
                                <button class="btn btn-sm btn-danger" onclick="stopCamera('${cameraName}')" title="Stop Recording"><i class="bi bi-square-fill"></i> Stop</button>
                            </div>
                        </div>
                    </div>
                </div>`;
            
            const canvas = document.getElementById('camera-' + cameraName);
            if (canvas) {
                const ctx = canvas.getContext("2d");
                const placeholderImage = new Image();
                placeholderImage.src = "assets/noConnection.png"; // Ensure this asset exists
                placeholderImage.onload = () => ctx.drawImage(placeholderImage, 0, 0, canvas.width, canvas.height);
                placeholderImage.onerror = () => { // If placeholder itself fails
                    ctx.fillStyle = "#404040"; ctx.fillRect(0,0,canvas.width,canvas.height);
                    ctx.fillStyle = "white"; ctx.textAlign="center"; ctx.font="12px Arial"; ctx.fillText("No Preview", canvas.width/2, canvas.height/2);
                }
            }
        }
    } catch (error) {
        console.error("Error loading camera HTML:", error);
        showErrorOnRecordPage(`Failed to load camera list: ${error.message || error}`);
        container.innerHTML = "<div class='col'><p class='text-danger text-center mt-3'>Error loading cameras.</p></div>";
    }
}

/** Sets the icon and behavior for the "Record All / Stop All" floating action button. */
function setRecordAllIcon(isAnyRecording) {
    const cameraFabButton = document.querySelector('.fab-container-right .fab[onclick*="startAllCameras"], .fab-container-right .fab[onclick*="stopAllCameras"]');
    if (!cameraFabButton) return;
    const cameraFabIcon = cameraFabButton.querySelector('i');
    if (!cameraFabIcon) return;

    if (isAnyRecording) {
        cameraFabIcon.classList.remove('bi-camera-video-fill');
        cameraFabIcon.classList.add('bi-square-fill');
        cameraFabIcon.style.color = 'red';
        cameraFabButton.onclick = stopAllCameras;
        cameraFabButton.title = "Stop All Recordings";
    } else {
        cameraFabIcon.classList.remove('bi-square-fill');
        cameraFabIcon.classList.add('bi-camera-video-fill');
        cameraFabIcon.style.color = 'white';
        cameraFabButton.onclick = startAllCameras;
        cameraFabButton.title = "Start All Recordings";
    }
}

/** Exposed to Python: Updates a camera's canvas with a new image blob. */
eel.expose(updateImageSrc);
async function updateImageSrc(name, val) {
    const canvas = document.getElementById('camera-' + name);
    if (!canvas) { return; }
    const ctx = canvas.getContext("2d");

    if (!val) { // Handle null or empty blob (e.g., error fetching thumbnail)
        console.warn(`Received null/empty image data for camera ${name}. Displaying placeholder.`);
        const placeholderImg = new Image();
        placeholderImg.src = "assets/noConnection.png";
        placeholderImg.onload = () => { ctx.clearRect(0,0,canvas.width,canvas.height); ctx.drawImage(placeholderImg,0,0,canvas.width,canvas.height); };
        placeholderImg.onerror = () => { ctx.clearRect(0,0,canvas.width,canvas.height); ctx.fillStyle="#404040"; ctx.fillRect(0,0,canvas.width,canvas.height); ctx.fillStyle="white"; ctx.fillText("No Preview", canvas.width/2, canvas.height/2);};
        return;
    }

    const img = new Image();
    img.src = "data:image/jpeg;base64," + val;
    canvas.setAttribute("cbas_image_source", img.src);

    img.onload = async function () {
        try {
            const settings = await eel.get_camera_settings(name)();
            if (!settings) {
                console.warn(`Settings not found for ${name} during thumbnail update.`);
                ctx.clearRect(0,0,canvas.width, canvas.height); ctx.fillStyle="#404040"; ctx.fillRect(0,0,canvas.width,canvas.height);
                ctx.fillStyle="white"; ctx.textAlign="center"; ctx.fillText("Settings Error", canvas.width/2, canvas.height/2);
                return;
            }
            const s = {
                x: Math.max(0, Math.min(1, parseFloat(settings['crop_left_x']))),
                y: Math.max(0, Math.min(1, parseFloat(settings['crop_top_y']))),
                w: Math.max(0, Math.min(1, parseFloat(settings['crop_width']))),
                h: Math.max(0, Math.min(1, parseFloat(settings['crop_height'])))
            };
            s.w = Math.min(s.w, 1 - s.x); s.h = Math.min(s.h, 1 - s.y);

            const cropX = s.x * img.naturalWidth; const cropY = s.y * img.naturalHeight;
            const cropWidth = s.w * img.naturalWidth; const cropHeight = s.h * img.naturalHeight;
            const targetResolution = parseInt(settings['resolution']) || Math.min(canvas.width, canvas.height);

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (cropWidth > 0 && cropHeight > 0) {
                drawImageOnCanvas(img, ctx, cropX, cropY, cropWidth, cropHeight, targetResolution);
            } else {
                 drawImageOnCanvas(img, ctx, 0, 0, img.naturalWidth, img.naturalHeight, targetResolution);
            }
        } catch (error) {
            console.error(`Error in updateImageSrc onload for ${name}:`, error);
             ctx.clearRect(0,0,canvas.width, canvas.height); ctx.fillStyle="#404040"; ctx.fillRect(0,0,canvas.width,canvas.height);
             ctx.fillStyle="white"; ctx.textAlign="center"; ctx.fillText("Display Error", canvas.width/2, canvas.height/2);
        }
    };
    img.onerror = function() {
        console.error(`Failed to load image data for camera ${name}.`);
        ctx.clearRect(0,0,canvas.width, canvas.height); ctx.fillStyle="#404040"; ctx.fillRect(0,0,canvas.width,canvas.height);
        ctx.fillStyle="white"; ctx.textAlign="center"; ctx.fillText("Image Load Fail", canvas.width/2, canvas.height/2);
    }
}

/** Updates UI buttons based on active recording streams. */
async function updateCamButtons() {
    try {
        const camerasResult = await eel.get_camera_list()();
        if (!camerasResult) { setRecordAllIcon(false); return; }
        const cameraNames = camerasResult.map(([name, _]) => name);
        const activeStreams = await eel.get_active_streams()() || [];
        setRecordAllIcon(activeStreams.length > 0);

        cameraNames.forEach(camName => {
            const beforeRec = document.getElementById(`before-recording-${camName}`);
            const duringRec = document.getElementById(`during-recording-${camName}`);
            if (beforeRec && duringRec) {
                const isCamActive = activeStreams.includes(camName);
                beforeRec.style.display = isCamActive ? 'none' : 'flex'; // Use display none/flex
                duringRec.style.display = isCamActive ? 'flex' : 'none';
            }
        });
    } catch (error) { console.error("Error updating camera buttons:", error); }
}

/** Loads initial camera list and triggers thumbnail downloads. */
async function loadCameras() {
    await loadCameraHTMLCards();
    await updateCamButtons();
    if (typeof eel !== 'undefined' && eel.download_camera_thumbnails) {
        eel.download_camera_thumbnails()();
    }
}

/** Draws a (potentially cropped) image scaled to fit a target resolution within the canvas. */
function drawImageOnCanvas(img, ctx, sx, sy, sw, sh, targetResolution) {
    const canvas = ctx.canvas;
    if (sw <= 0 || sh <= 0) { ctx.clearRect(0, 0, canvas.width, canvas.height); return; }
    const ratio = Math.min(targetResolution / sw, targetResolution / sh);
    const drawWidth = sw * ratio;
    const drawHeight = sh * ratio;
    const destX = (canvas.width - drawWidth) / 2;
    const destY = (canvas.height - drawHeight) / 2;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, sx, sy, sw, sh, destX, destY, drawWidth, drawHeight);
}

/** Loads camera settings into the modal for editing. */
async function loadCameraSettings(cameraName) {
    const modalCanvas = document.getElementById("camera-image-settings-preview"); // Updated ID
    if (!modalCanvas) { console.error("Modal canvas 'camera-image-settings-preview' not found."); return; }
    const modalCtx = modalCanvas.getContext("2d");

    try {
        const settings = await eel.get_camera_settings(cameraName)();
        if (settings) {
            originalCameraNameForSettings = cameraName;
            document.getElementById('cs-name').value = cameraName;
            document.getElementById('cs-url').value = settings['rtsp_url'];
            document.getElementById('cs-framerate').value = settings['framerate'];
            document.getElementById('cs-resolution').value = settings['resolution'];
            document.getElementById('cs-cropx').value = settings['crop_left_x'];
            document.getElementById('cs-cropy').value = settings['crop_top_y'];
            document.getElementById('cs-crop-width').value = settings['crop_width'];
            document.getElementById('cs-crop-height').value = settings['crop_height'];

            const fieldsToWatch = ['cs-resolution', 'cs-cropx', 'cs-cropy', 'cs-crop-width', 'cs-crop-height'];
            fieldsToWatch.forEach(id => {
                const el = document.getElementById(id);
                if (el) el.oninput = () => drawBoundsOnModalCanvas(modalImage); // Use global modalImage
            });
            
            const mainCanvasImageSrc = document.getElementById('camera-' + cameraName)?.getAttribute("cbas_image_source");
            modalPreviewImage.onload = () => { // Use the global modalPreviewImage
                const aspectRatio = modalPreviewImage.naturalWidth / modalPreviewImage.naturalHeight;
                const modalCanvasMaxWidth = 400; 
                modalCanvas.width = Math.min(modalPreviewImage.naturalWidth, modalCanvasMaxWidth);
                modalCanvas.height = modalCanvas.width / aspectRatio;
                drawBoundsOnModalCanvas(modalPreviewImage);
            };
            modalPreviewImage.onerror = () => {
                 modalCtx.clearRect(0,0,modalCanvas.width, modalCanvas.height);
                 modalCtx.fillStyle="grey"; modalCtx.fillRect(0,0,modalCanvas.width,modalCanvas.height);
                 modalCtx.fillStyle="white"; modalCtx.textAlign="center"; modalCtx.fillText("Preview Error", modalCanvas.width/2, modalCanvas.height/2);
            };
            modalPreviewImage.src = mainCanvasImageSrc || "assets/noConnection.png";


            if (cameraSettingsBsModal) cameraSettingsBsModal.show();
        } else {
            showErrorOnRecordPage(`Could not load settings for camera: ${cameraName}`);
        }
    } catch (error) {
        console.error(`Error in loadCameraSettings for ${cameraName}:`, error);
        showErrorOnRecordPage(`Failed to load settings: ${error.message || error}`);
    }
}

/** Redraws the crop bounds on the modal's canvas. */
function drawBoundsOnModalCanvas(imageToDraw) { // Pass the image object
    const modalCanvas = document.getElementById("camera-image-settings-preview");
    if (!modalCanvas || !imageToDraw || !imageToDraw.complete || imageToDraw.naturalWidth === 0) {
        return;
    }
    const modalCtx = modalCanvas.getContext("2d");

    let cxVal = Math.max(0, Math.min(1, parseFloat(document.getElementById('cs-cropx').value) || 0));
    let cyVal = Math.max(0, Math.min(1, parseFloat(document.getElementById('cs-cropy').value) || 0));
    let cwVal = Math.max(0, parseFloat(document.getElementById('cs-crop-width').value) || 1);
    let chVal = Math.max(0, parseFloat(document.getElementById('cs-crop-height').value) || 1);
    cwVal = Math.min(cwVal, 1 - cxVal); chVal = Math.min(chVal, 1 - cyVal);
    
    document.getElementById('cs-cropx').value = cxVal.toFixed(3);
    document.getElementById('cs-cropy').value = cyVal.toFixed(3);
    document.getElementById('cs-crop-width').value = cwVal.toFixed(3);
    document.getElementById('cs-crop-height').value = chVal.toFixed(3);

    const sx = cxVal * imageToDraw.naturalWidth; const sy = cyVal * imageToDraw.naturalHeight;
    const sw = cwVal * imageToDraw.naturalWidth; const sh = chVal * imageToDraw.naturalHeight;

    modalCtx.clearRect(0, 0, modalCanvas.width, modalCanvas.height);
    // Draw the image scaled to fit the canvas first
    const canvasAspect = modalCanvas.width / modalCanvas.height;
    const imgAspect = imageToDraw.naturalWidth / imageToDraw.naturalHeight;
    let drawW, drawH, drawX, drawY;
    if (imgAspect > canvasAspect) { // Image wider than canvas
        drawW = modalCanvas.width;
        drawH = modalCanvas.width / imgAspect;
        drawX = 0;
        drawY = (modalCanvas.height - drawH) / 2;
    } else { // Image taller or same aspect
        drawH = modalCanvas.height;
        drawW = modalCanvas.height * imgAspect;
        drawY = 0;
        drawX = (modalCanvas.width - drawW) / 2;
    }
    modalCtx.drawImage(imageToDraw, drawX, drawY, drawW, drawH);
    
    if (sw > 0 && sh > 0) {
        modalCtx.strokeStyle = 'rgba(255, 0, 0, 0.7)';
        modalCtx.lineWidth = Math.max(1, modalCanvas.width / 150); // Scale line width
        modalCtx.strokeRect(
            drawX + (sx * (drawW / imageToDraw.naturalWidth)), 
            drawY + (sy * (drawH / imageToDraw.naturalHeight)), 
            sw * (drawW / imageToDraw.naturalWidth), 
            sh * (drawH / imageToDraw.naturalHeight)
        );
    }
}

/** Saves updated camera settings. */
async function saveCameraSettings() {
    const newCameraName = document.getElementById('cs-name').value;
    if (!newCameraName.trim()) { showErrorOnRecordPage("Camera name cannot be empty."); return; }
    const settingsToSave = {
        "rtsp_url": document.getElementById('cs-url').value,
        "framerate": parseInt(document.getElementById('cs-framerate').value) || 10,
        "resolution": parseInt(document.getElementById('cs-resolution').value) || 256,
        'crop_left_x': parseFloat(document.getElementById('cs-cropx').value) || 0,
        'crop_top_y': parseFloat(document.getElementById('cs-cropy').value) || 0,
        'crop_width': parseFloat(document.getElementById('cs-crop-width').value) || 1,
        'crop_height': parseFloat(document.getElementById('cs-crop-height').value) || 1,
    };
    try {
        let finalCameraNameToUse = originalCameraNameForSettings;
        if (newCameraName !== originalCameraNameForSettings) {
            const renameSuccess = await eel.rename_camera(originalCameraNameForSettings, newCameraName)();
            if (!renameSuccess) { showErrorOnRecordPage(`Failed to rename. '${newCameraName}' may exist.`); return; }
            finalCameraNameToUse = newCameraName;
        }
        const saveSuccess = await eel.save_camera_settings(finalCameraNameToUse, settingsToSave)();
        if (saveSuccess) {
            if (cameraSettingsBsModal) cameraSettingsBsModal.hide();
            await loadCameras(); 
        } else { showErrorOnRecordPage("Failed to save settings (backend error)."); }
    } catch (e) { console.error("Save settings error:", e); showErrorOnRecordPage(`Error: ${e.message||e}`); }
}

/** Shows the "Add Camera" modal. */
function showAddCameraModal() { if (addCameraBsModal) addCameraBsModal.show(); } // Renamed for clarity

/** Shows the status modal. */
async function showStatusModal() {
    try {
        const status = await eel.get_cbas_status()();
        const streamsEl = document.getElementById("status-streams");
        const encodeEl = document.getElementById("status-encode-count");
        if(status){
            if (streamsEl) streamsEl.innerText = (status.streams) ? "Recording: " + status.streams.join(", ") : "No cameras recording.";
            if (encodeEl) encodeEl.innerText = "Files to encode: " + status.encode_file_count;
        } else {
            if(streamsEl) streamsEl.innerText = "Status unavailable.";
            if(encodeEl) encodeEl.innerText = "Files to encode: unavailable";
        }
        if (statusBsModal) statusBsModal.show();
    } catch (e) { console.error("Get status error:", e); }
}

/** Updates the animated status icon in the FAB. */
async function updateStatusIcon() { // Renamed for clarity
    const camIcon = document.getElementById("status-camera-icon");
    const camOutline = document.getElementById("status-camera-outline");
    if (!camIcon || !camOutline) return;
    try {
        const status = await eel.get_cbas_status()();
        const isStreaming = status && status.streams && status.streams.length > 0;
        const animationClass = "status-blinking"; 
        [camIcon, camOutline].forEach(el => {
            el.style.color = isStreaming ? "red" : "white";
            isStreaming ? el.classList.add(animationClass) : el.classList.remove(animationClass);
        });
    } catch (e) { console.error("Update status icon error:", e); }
}

/** Adds a new camera. */
async function addCameraSubmit() { // Renamed for clarity
    const name = document.getElementById('camera-name-modal-input').value; // Assuming different ID for modal input
    const rtsp = document.getElementById('rtsp-url-modal-input').value;
    if (!name.trim() || !rtsp.trim()) { showErrorOnRecordPage('Name and RTSP URL required.'); return; }
    try {
        const success = await eel.create_camera(name, rtsp)();
        if (success) {
            if (addCameraBsModal) addCameraBsModal.hide();
            document.getElementById('camera-name-modal-input').value = ""; 
            document.getElementById('rtsp-url-modal-input').value = "";
            await loadCameras();
        } else { showErrorOnRecordPage(`Failed to create '${name}'. It may exist.`); }
    } catch (e) { console.error("Add camera error:", e); showErrorOnRecordPage("Error adding camera."); }
}

/** Opens VLC for live view. */
function liveViewCamera(cameraName) { eel.open_camera_live_view(cameraName)(); }

/** Starts recording for a single camera. */
async function startCamera(cameraName) {
    try {
        const dir = await eel.create_recording_dir(cameraName)();
        if (!dir) { showErrorOnRecordPage('Could not create recording directory for ' + cameraName); return; }
        const success = await eel.start_camera_stream(cameraName, dir, 600)();
        if (success) {
            document.getElementById(`before-recording-${cameraName}`).style.display = 'none';
            document.getElementById(`during-recording-${cameraName}`).style.display = 'flex';
        } else { showErrorOnRecordPage(`'${cameraName}' might be recording or error occurred.`); }
        await updateStatusIcon();
    } catch (e) { console.error(`Start ${cameraName} error:`, e); }
}

/** Stops recording for a single camera. */
async function stopCamera(cameraName) {
    try {
        const success = await eel.stop_camera_stream(cameraName)();
        if (success) {
            document.getElementById(`during-recording-${cameraName}`).style.display = 'none';
            document.getElementById(`before-recording-${cameraName}`).style.display = 'flex';
        } else { showErrorOnRecordPage(`Could not stop ${cameraName}. Not recording?`); }
        await updateStatusIcon();
    } catch (e) { console.error(`Stop ${cameraName} error:`, e); }
}

/** Starts all cameras. */
async function startAllCameras() {
    try {
        const cameras = await eel.get_camera_list()();
        if (cameras && cameras.length > 0) {
            for (const [name, _] of cameras) await startCamera(name); // Sequential
            await updateCamButtons(); // Update all buttons after attempting all starts
        } else { alert("No cameras configured."); }
    } catch (e) { console.error("Start all error:", e); }
}

/** Stops all cameras. */
async function stopAllCameras() {
    try {
        const cameras = await eel.get_camera_list()();
        if (cameras && cameras.length > 0) {
            for (const [name, _] of cameras) await stopCamera(name); // Sequential
            await updateCamButtons();
        } else { alert("No cameras configured."); }
    } catch (e) { console.error("Stop all error:", e); }
}

// Initial Page Load
document.addEventListener('DOMContentLoaded', () => {
    loadCameras();
    setInterval(updateStatusIcon, 3000); 
    // Assign event listener to the "Add Camera" submit button in the modal
    const addCamSubmitBtn = document.getElementById('addCameraButton'); // Assuming this ID for the submit button in addCamera modal
    if(addCamSubmitBtn) addCamSubmitBtn.onclick = addCameraSubmit;
});

// Unload Handlers
window.addEventListener("unload", () => {
    if (!routingInProgress) { if (typeof eel !== 'undefined' && eel.kill_streams) eel.kill_streams()().catch(err => console.error("Unload kill_streams error:", err)); }
});
window.onbeforeunload = () => {
    if (!routingInProgress) { if (typeof eel !== 'undefined' && eel.kill_streams) eel.kill_streams()().catch(err => console.error("Beforeunload kill_streams error:", err)); }
    // return null; 
};

/** Exposed to Python: Updates the inference progress bar. */
eel.expose(inferLoadBar);
function inferLoadBar(progresses) { 
    const barContainer = document.getElementById('inference-bar-container'); 
    const barElement = document.getElementById('inference-bar');
    if (!barContainer || !barElement) return;

    if (progresses && Array.isArray(progresses) && progresses.length > 0) {
        barContainer.style.visibility = 'visible';
        barElement.innerHTML = ''; 

        progresses.forEach((progressValue) => {
            const segment = document.createElement('div');
            segment.classList.add('progress-bar', 'progress-bar-striped', 'progress-bar-animated');
            let widthPercent = Math.max(0, Math.min(100, progressValue)); // Clamp between 0 and 100
            let bgColorClass = 'bg-success';
            if (progressValue < 0) { // Error
                bgColorClass = 'bg-danger';
                widthPercent = (1 / progresses.length) * 100; // Error takes up its share of space
            }
            segment.classList.add(bgColorClass);
            segment.style.width = `${widthPercent}%`;
            segment.setAttribute('aria-valuenow', widthPercent);
            barElement.appendChild(segment);
        });
    } else {
        barContainer.style.visibility = 'hidden';
        barElement.innerHTML = '';
    }
}