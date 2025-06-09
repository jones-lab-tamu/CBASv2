/**
 * @file visualize_page.js
 * Handles the visualization page UI, primarily for generating and displaying actograms.
 */

// const ipc = window.ipcRenderer; // Not used in this file

/** Flag to prevent `kill_streams` from being called during intentional page navigation. */
let routingInProgress = false;

// General error modal (assuming one errorModal element is shared across pages or defined in HTML)
const errorModalElement = document.getElementById("errorModal");
const errorMessageElement = document.getElementById("error-message");
let generalErrorBsModal = errorModalElement ? new bootstrap.Modal(errorModalElement) : null;

/** Shows the error modal with a custom message. */
function showErrorOnVisualizePage(message) {
    if (errorMessageElement && generalErrorBsModal) {
        errorMessageElement.innerText = message;
        generalErrorBsModal.show();
    } else {
        alert(message); // Fallback
    }
}


// --- Routing Functions ---
function routeToRecordPage() { routingInProgress = true; window.location.href = './record.html';}
function routeToLabelTrainPage() { routingInProgress = true; window.location.href = './label-train.html';}
function routeToVisualizePage() { routingInProgress = true; window.location.href = './visualize.html';}

// --- Eel Exposed Functions (Python to JS) ---

/**
 * Exposed to Python: Updates the actogram image display.
 * @param {string | null} base64Val - Base64 encoded image string, or null if error/no image.
 */
eel.expose(updateActogramDisplay);
function updateActogramDisplay(base64Val) {
    const imageElement = document.getElementById('actogram-image');
    const loadingSpinnerElement = document.getElementById('loading-spinner-actogram'); 

    if (imageElement) {
        if (base64Val) {
            imageElement.src = "data:image/jpeg;base64," + base64Val;
            imageElement.style.display = "block";
        } else {
            // Use the correct placeholder for this page
            imageElement.src = "assets/noData.png"; 
            imageElement.style.display = "block"; 
            console.log("updateActogramDisplay received null, showing placeholder.");
        }
    }
    if (loadingSpinnerElement) loadingSpinnerElement.style.display = "none"; 
}

// --- UI Helper Functions ---

/** Shows a loading spinner specific to the actogram area. */
function showActogramLoadingIndicator() {
    const spinner = document.getElementById('loading-spinner-actogram');
    if (spinner) spinner.style.display = "block";
    const actogramImage = document.getElementById('actogram-image');
    if (actogramImage) actogramImage.style.display = "none"; 
}

/**
 * Toggles the visibility of an HTML element (used for an accordion-style tree).
 * @param {string} elementId - The ID of the element to toggle.
 */
function toggleVisibility(elementId) {
    const elem = document.getElementById(elementId);
    if (elem) {
        elem.style.display = (elem.style.display === 'none' || elem.style.display === '') ? 'block' : 'none';
    }
}

// --- Actogram Generation and Adjustment ---

/**
 * Gathers parameters from the UI and calls the backend to generate a new actogram.
 */
async function generateAndDisplayActogram(rootDir, sessionDir, modelName, behaviorName) {
    const framerateStr = document.getElementById('vs-framerate').value;
    const binsizeMinutesStr = document.getElementById('vs-binsize').value;
    const startTimeStr = document.getElementById('vs-start').value;
    const thresholdPercentStr = document.getElementById('vs-threshold').value;
    const lightCyclePatternStr = document.getElementById('vs-lcycle').value;
	const plotAcrophase = document.getElementById('vs-acrophase').checked; // Get checkbox state

    // Corrected validation check
    if (!framerateStr || !binsizeMinutesStr || !startTimeStr || !thresholdPercentStr) {
        showErrorOnVisualizePage("All actogram parameters must be filled.");
        return;
    }

    const actogramTitleElement = document.getElementById('actogram-title');
    if (actogramTitleElement) {
        actogramTitleElement.textContent = `Actogram: ${modelName} - ${behaviorName} (Recording: ${sessionDir})`;
    }
    
    showActogramLoadingIndicator(); 

    try {
        // Corrected function call (removed color and norm)
        await eel.make_actogram(
            rootDir, sessionDir, modelName, behaviorName,
            framerateStr, binsizeMinutesStr, startTimeStr,
            thresholdPercentStr, lightCyclePatternStr, plotAcrophase
        )();
    } catch (error) {
        console.error("Error calling eel.make_actogram:", error);
        updateActogramDisplay(null);
        showErrorOnVisualizePage(`Failed to generate actogram: ${error.message || error}`);
    }
}

/** Gathers parameters from the UI and calls the backend to adjust the current actogram. */
async function adjustCurrentDisplayActogram() {
    const framerateStr = document.getElementById('vs-framerate').value;
    const binsizeMinutesStr = document.getElementById('vs-binsize').value;
    const startTimeStr = document.getElementById('vs-start').value;
    const thresholdPercentStr = document.getElementById('vs-threshold').value;
    const lightCyclePatternStr = document.getElementById('vs-lcycle').value;
	const plotAcrophase = document.getElementById('vs-acrophase').checked;
    
    // Corrected validation check
    if (!framerateStr || !binsizeMinutesStr || !startTimeStr || !thresholdPercentStr) {
        showErrorOnVisualizePage("All actogram parameters must be filled for adjustment.");
        return;
    }

    showActogramLoadingIndicator();
    try {
        // Corrected function call (removed color and norm)
        await eel.adjust_actogram(
            framerateStr, binsizeMinutesStr, startTimeStr,
            thresholdPercentStr, lightCyclePatternStr, plotAcrophase
        )();
    } catch (error) {
        console.error("Error calling eel.adjust_actogram:", error);
        updateActogramDisplay(null);
        showErrorOnVisualizePage(`Failed to adjust actogram: ${error.message || error}`);
    }
}

/** Initializes the page by fetching the recording tree and populating the selection UI. */
async function initializeVisualizePageContent() {
    const directoriesContainer = document.getElementById('directories');
    if (!directoriesContainer) { console.error("Directories container not found!"); return;}

    try {
        const recordingTree = await eel.get_recording_tree()();
        if (!recordingTree || recordingTree.length === 0) {
            showErrorOnVisualizePage('No recordings with classifications found. Please run inference on some recordings first.');
            directoriesContainer.innerHTML = "<p class='text-light'>No classified recordings available to visualize.</p>";
            return;
        }

        let htmlBuilder = '';
        recordingTree.forEach((dateEntry) => {
            const dateStr = dateEntry[0];
            const sessions = dateEntry[1];
            const dateId = `rd-${dateStr.replace(/[\s\/\\:]/g, '-')}`;

            htmlBuilder += `<h5 class='text-light mt-2 hand-cursor' onclick="toggleVisibility('${dateId}')"><i class="bi bi-calendar-date-fill me-2"></i>${dateStr}</h5>`;
            htmlBuilder += `<div id='${dateId}' class='ms-3' style="display:none;">`;

            sessions.forEach((sessionEntry) => {
                const sessionName = sessionEntry[0];
                const models = sessionEntry[1];
                const sessionId = `${dateId}-sd-${sessionName.replace(/[\s\/\\:]/g, '-')}`;

                htmlBuilder += `<h6 class='text-light mt-1 hand-cursor' onclick="toggleVisibility('${sessionId}')"><i class="bi bi-camera-reels-fill me-2"></i>${sessionName}</h6>`;
                htmlBuilder += `<div id='${sessionId}' class='ms-3' style="display:none;">`;

                models.forEach((modelEntry) => {
                    const modelName = modelEntry[0];
                    const behaviors = modelEntry[1];
                    const modelId = `${sessionId}-md-${modelName.replace(/[\s\/\\:]/g, '-')}`;

                    htmlBuilder += `<div class='text-light mt-1 hand-cursor' onclick="toggleVisibility('${modelId}')"><i class="bi bi-cpu-fill me-2"></i>${modelName}</div>`;
                    htmlBuilder += `<div id='${modelId}' class='ms-3' style="display:none;">`;
                    
                    behaviors.forEach((behaviorName) => {
                        const safeBehaviorNameForId = behaviorName.replace(/[\s\/\\:]/g, '-');
                        const behaviorId = `${modelId}-beh-${safeBehaviorNameForId}`; 
                        htmlBuilder += `
                            <div class="form-check my-1">
                                <input class="form-check-input" type="radio" name="actogramSelectionRadio" id="${behaviorId}" 
                                       onclick="generateAndDisplayActogram('${dateStr}','${sessionName}','${modelName}','${behaviorName}')">
                                <label class="form-check-label text-light small" for="${behaviorId}">
                                    ${behaviorName}
                                </label>
                            </div>`;
                    });
                    htmlBuilder += `</div>`; 
                });
                htmlBuilder += `</div>`; 
            });
            htmlBuilder += `</div>`; 
        });
        directoriesContainer.innerHTML = htmlBuilder;

    } catch (error) {
        console.error("Error initializing visualization page:", error);
        if (directoriesContainer) directoriesContainer.innerHTML = "<p class='text-danger text-center'>Error loading recording data. Check console.</p>";
        showErrorOnVisualizePage(`Error loading data: ${error.message || error}`);
    }
}

// --- Event Listeners ---

/** Handles page unload. */
window.addEventListener("unload", function(){
    if(!routingInProgress) { if (typeof eel !== 'undefined' && eel.kill_streams) eel.kill_streams()().catch(err => console.error("Unload kill_streams error:", err));}
});
window.onbeforeunload = function (){
    if(!routingInProgress) { if (typeof eel !== 'undefined' && eel.kill_streams) eel.kill_streams()().catch(err => console.error("Beforeunload kill_streams error:", err));}
};

/**
 * Waits for the Eel WebSocket to be fully connected before proceeding.
 */
function waitForEelConnection() {
    return new Promise(resolve => {
        if (eel._websocket && eel._websocket.readyState === 1) {
            resolve();
            return;
        }
        const interval = setInterval(() => {
            if (eel._websocket && eel._websocket.readyState === 1) {
                clearInterval(interval);
                resolve();
            }
        }, 100);
    });
}

// Initial load of the page content
document.addEventListener('DOMContentLoaded', async () => {
    await waitForEelConnection();
    initializeVisualizePageContent();

    const adjustmentControlsIds = ['vs-framerate', 'vs-binsize', 'vs-start', 'vs-threshold', 'vs-lcycle', 'vs-acrophase'];
    adjustmentControlsIds.forEach(controlId => {
        const elem = document.getElementById(controlId);
        if (elem) {
            // Use 'change' for select dropdowns, and 'input' for sliders/number inputs for responsiveness
            const eventType = (elem.type === 'checkbox' || elem.tagName.toLowerCase() === 'select') ? 'change' : 'input';
            elem.addEventListener(eventType, adjustCurrentDisplayActogram);
        }
    });
});