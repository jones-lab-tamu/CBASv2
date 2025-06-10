/**
 * @file Manages the Visualize page UI and interactions.
 * @description This file handles building the data selection tree, generating actograms
 * based on user input, and adjusting actogram parameters in real-time.
 */

// =================================================================
// GLOBAL STATE & VARIABLES
// =================================================================

/** Flag to prevent `kill_streams` from being called during intentional page navigation. */
let routingInProgress = false;

// --- Bootstrap Modal Instance ---
const errorModalElement = document.getElementById("errorModal");
let generalErrorBsModal = errorModalElement ? new bootstrap.Modal(errorModalElement) : null;


// =================================================================
// ROUTING & UTILITY FUNCTIONS
// =================================================================

function routeToRecordPage() { routingInProgress = true; window.location.href = './record.html'; }
function routeToLabelTrainPage() { routingInProgress = true; window.location.href = './label-train.html'; }
function routeToVisualizePage() { routingInProgress = true; window.location.href = './visualize.html'; }

/** Shows the general error modal with a custom message. */
function showErrorOnVisualizePage(message) {
    const errorMessageElement = document.getElementById("error-message");
    if (errorMessageElement && generalErrorBsModal) {
        errorMessageElement.innerText = message;
        generalErrorBsModal.show();
    } else {
        alert(message); // Fallback
    }
}

/**
 * Toggles the visibility of an HTML element, used for the accordion-style selection tree.
 * @param {string} elementId - The ID of the element to toggle.
 */
function toggleVisibility(elementId) {
    const elem = document.getElementById(elementId);
    if (elem) {
        elem.style.display = (elem.style.display === 'none' || elem.style.display === '') ? 'block' : 'none';
    }
}

/** Shows a loading spinner in the actogram display area. */
function showActogramLoadingIndicator() {
    const spinner = document.getElementById('loading-spinner-actogram');
    if (spinner) spinner.style.display = "block";
    const actogramImage = document.getElementById('actogram-image');
    if (actogramImage) actogramImage.style.display = "none";
}


// =================================================================
// EEL-EXPOSED FUNCTIONS (Called FROM Python)
// =================================================================

/**
 * Updates the actogram image display with a new base64 encoded image from the backend.
 * @param {string | null} base64Val - Base64 encoded image string, or null on error.
 */
eel.expose(updateActogramDisplay);
function updateActogramDisplay(base64Val) {
    const imageElement = document.getElementById('actogram-image');
    const loadingSpinnerElement = document.getElementById('loading-spinner-actogram');

    if (imageElement) {
        imageElement.src = base64Val ? "data:image/jpeg;base64," + base64Val : "assets/noData.png";
        imageElement.style.display = "block";
    }
    if (loadingSpinnerElement) {
        loadingSpinnerElement.style.display = "none";
    }
}


// =================================================================
// CORE APPLICATION LOGIC
// =================================================================

/**
 * Gathers parameters from the UI and calls the backend to generate a new actogram.
 * This is triggered when a user selects a specific behavior radio button.
 * @param {string} rootDir - The date directory of the recording.
 * @param {string} sessionDir - The session directory of the recording.
 * @param {string} modelName - The name of the model to use.
 * @param {string} behaviorName - The name of the behavior to plot.
 */
async function generateAndDisplayActogram(rootDir, sessionDir, modelName, behaviorName) {
    const framerate = document.getElementById('vs-framerate').value;
    const binsize = document.getElementById('vs-binsize').value;
    const start = document.getElementById('vs-start').value;
    const threshold = document.getElementById('vs-threshold').value;
    const lightcycle = document.getElementById('vs-lcycle').value;
    const plotAcrophase = document.getElementById('vs-acrophase').checked;

    if (!framerate || !binsize || !start || !threshold) {
        showErrorOnVisualizePage("All actogram parameters must be filled.");
        return;
    }

    const titleElement = document.getElementById('actogram-title');
    if (titleElement) {
        titleElement.textContent = `Actogram: ${modelName} - ${behaviorName} (Recording: ${sessionDir})`;
    }
    
    showActogramLoadingIndicator();

    try {
        await eel.make_actogram(
            rootDir, sessionDir, modelName, behaviorName,
            framerate, binsize, start, threshold, lightcycle, plotAcrophase
        )();
    } catch (error) {
        console.error("Error calling eel.make_actogram:", error);
        updateActogramDisplay(null);
        showErrorOnVisualizePage(`Failed to generate actogram: ${error.message || error}`);
    }
}

/**
 * Gathers parameters from the UI and calls the backend to adjust the current actogram.
 * This is triggered by the 'onchange' or 'oninput' event of the control fields.
 */
async function adjustCurrentDisplayActogram() {
    const framerate = document.getElementById('vs-framerate').value;
    const binsize = document.getElementById('vs-binsize').value;
    const start = document.getElementById('vs-start').value;
    const threshold = document.getElementById('vs-threshold').value;
    const lightcycle = document.getElementById('vs-lcycle').value;
    const plotAcrophase = document.getElementById('vs-acrophase').checked;
    
    if (!framerate || !binsize || !start || !threshold) {
        // Silently return on adjustment to avoid spamming user with errors while typing.
        return;
    }

    showActogramLoadingIndicator();
    try {
        await eel.adjust_actogram(
            framerate, binsize, start, threshold, lightcycle, plotAcrophase
        )();
    } catch (error) {
        console.error("Error calling eel.adjust_actogram:", error);
        updateActogramDisplay(null);
        showErrorOnVisualizePage(`Failed to adjust actogram: ${error.message || error}`);
    }
}

/**
 * Initializes the page by fetching the recording tree from Python and building the selection UI.
 */
async function initializeVisualizePageContent() {
    const container = document.getElementById('directories');
    if (!container) return;

    try {
        const recordingTree = await eel.get_recording_tree()();
        if (!recordingTree || recordingTree.length === 0) {
            container.innerHTML = "<p class='text-light'>No classified recordings available to visualize.</p>";
            showErrorOnVisualizePage('No classified recordings found. Please run inference on some videos first.');
            return;
        }

        let htmlBuilder = '';
        recordingTree.forEach((dateEntry) => {
            const [dateStr, sessions] = dateEntry;
            const dateId = `rd-${dateStr.replace(/[\W_]+/g, '-')}`;
            htmlBuilder += `<h5 class='text-light mt-2 hand-cursor' onclick="toggleVisibility('${dateId}')"><i class="bi bi-calendar-date-fill me-2"></i>${dateStr}</h5>`;
            htmlBuilder += `<div id='${dateId}' class='ms-3' style="display:none;">`;

            sessions.forEach((sessionEntry) => {
                const [sessionName, models] = sessionEntry;
                const sessionId = `${dateId}-sd-${sessionName.replace(/[\W_]+/g, '-')}`;
                htmlBuilder += `<h6 class='text-light mt-1 hand-cursor' onclick="toggleVisibility('${sessionId}')"><i class="bi bi-camera-reels-fill me-2"></i>${sessionName}</h6>`;
                htmlBuilder += `<div id='${sessionId}' class='ms-3' style="display:none;">`;

                models.forEach((modelEntry) => {
                    const [modelName, behaviors] = modelEntry;
                    const modelId = `${sessionId}-md-${modelName.replace(/[\W_]+/g, '-')}`;
                    htmlBuilder += `<div class='text-info mt-1 hand-cursor' onclick="toggleVisibility('${modelId}')"><i class="bi bi-cpu-fill me-2"></i>${modelName}</div>`;
                    htmlBuilder += `<div id='${modelId}' class='ms-3' style="display:none;">`;
                    
                    behaviors.forEach((behaviorName) => {
                        const behaviorId = `${modelId}-beh-${behaviorName.replace(/[\W_]+/g, '-')}`;
                        htmlBuilder += `
                            <div class="form-check my-1">
                                <input class="form-check-input" type="radio" name="actogramSelectionRadio" id="${behaviorId}" onclick="generateAndDisplayActogram('${dateStr}','${sessionName}','${modelName}','${behaviorName}')">
                                <label class="form-check-label text-light small" for="${behaviorId}">${behaviorName}</label>
                            </div>`;
                    });
                    htmlBuilder += `</div>`;
                });
                htmlBuilder += `</div>`;
            });
            htmlBuilder += `</div>`;
        });
        container.innerHTML = htmlBuilder;

    } catch (error) {
        console.error("Error initializing visualization page:", error);
        container.innerHTML = "<p class='text-danger text-center'>Error loading recording data.</p>";
        showErrorOnVisualizePage(`Error loading data: ${error.message || error}`);
    }
}


// =================================================================
// PAGE INITIALIZATION & EVENT LISTENERS
// =================================================================

/** Waits for the Eel WebSocket to be fully connected before proceeding. */
function waitForEelConnection() {
    return new Promise(resolve => {
        if (eel._websocket && eel._websocket.readyState === 1) {
            resolve(); return;
        }
        const interval = setInterval(() => {
            if (eel._websocket && eel._websocket.readyState === 1) {
                clearInterval(interval);
                resolve();
            }
        }, 100);
    });
}

/** Main execution block that runs when the page is fully loaded. */
document.addEventListener('DOMContentLoaded', async () => {
    await waitForEelConnection();
    initializeVisualizePageContent();

    const adjustmentControlsIds = ['vs-framerate', 'vs-binsize', 'vs-start', 'vs-threshold', 'vs-lcycle', 'vs-acrophase'];
    adjustmentControlsIds.forEach(controlId => {
        const elem = document.getElementById(controlId);
        if (elem) {
            const eventType = (elem.type === 'checkbox' || elem.tagName.toLowerCase() === 'select') ? 'change' : 'input';
            elem.addEventListener(eventType, adjustCurrentDisplayActogram);
        }
    });
});

/** Page unload listeners to ensure Python processes are killed. */
window.addEventListener("unload", () => {
    if (!routingInProgress) { eel.kill_streams()?.().catch(err => console.error(err)); }
});
window.onbeforeunload = () => {
    if (!routingInProgress) { eel.kill_streams()?.().catch(err => console.error(err)); }
};