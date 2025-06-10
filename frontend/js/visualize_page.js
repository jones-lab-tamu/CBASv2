/**
 * @file Manages the Visualize page UI and interactions.
 * @description This file handles building the data selection tree, generating multiple
 * tiled actograms based on user input, and adjusting them in real-time.
 */

// =================================================================
// GLOBAL STATE & VARIABLES
// =================================================================

let routingInProgress = false;
const errorModalElement = document.getElementById("errorModal");
let generalErrorBsModal = errorModalElement ? new bootstrap.Modal(errorModalElement) : null;

let currentSelection = { root: null, session: null, model: null };

// --- Task & Debounce Management ---
let actogramDebounceTimer;
let latestVizTaskId = 0; // The ID of the most recent request sent from the frontend.

// =================================================================
// ROUTING & UTILITY FUNCTIONS
// =================================================================

function routeToRecordPage() { routingInProgress = true; window.location.href = './record.html'; }
function routeToLabelTrainPage() { routingInProgress = true; window.location.href = './label-train.html'; }
function routeToVisualizePage() { routingInProgress = true; window.location.href = './visualize.html'; }

function showErrorOnVisualizePage(message) {
    const errorMessageElement = document.getElementById("error-message");
    if (errorMessageElement && generalErrorBsModal) {
        errorMessageElement.innerText = message;
        generalErrorBsModal.show();
    } else { alert(message); }
}

function toggleVisibility(elementId) {
    const elem = document.getElementById(elementId);
    if (elem) elem.style.display = (elem.style.display === 'none' || elem.style.display === '') ? 'block' : 'none';
}

function showActogramLoadingIndicator() {
    const spinner = document.getElementById('loading-spinner-actogram');
    if (spinner) spinner.style.display = "block";
    const container = document.getElementById('actogram-container');
    if(container) container.innerHTML = ''; // Clear old actograms
    const placeholder = document.getElementById('actogram-placeholder');
    if (placeholder) placeholder.style.display = 'none';
}


// =================================================================
// EEL-EXPOSED FUNCTIONS (Called FROM Python)
// =================================================================

eel.expose(updateActogramDisplay);
function updateActogramDisplay(results, taskId) {
    // *** CRITICAL CHECK ***: Only update the UI if the returned data is from the most recent request.
    if (taskId !== latestVizTaskId) {
        console.log(`Ignoring obsolete UI update for task ${taskId}. Current task is ${latestVizTaskId}.`);
        return;
    }

    const container = document.getElementById('actogram-container');
    const spinner = document.getElementById('loading-spinner-actogram');
    const placeholder = document.getElementById('actogram-placeholder');

    if (spinner) spinner.style.display = "none";
    if (!container) return;

    container.innerHTML = '';
    if (results && results.length > 0) {
        if (placeholder) placeholder.style.display = 'none';
        let html = '';
        const colClass = results.length === 1 ? 'col-12' : 'col-xl-6'; // Full width if one, half if more

        results.forEach(result => {
            html += `
                <div class="${colClass}">
                    <div class="card bg-dark text-light">
                        <div class="card-header text-center">
                            <h6>${result.behavior}</h6>
                        </div>
                        <div class="card-body p-1">
                            <img src="data:image/png;base64,${result.blob}" class="img-fluid" alt="Actogram for ${result.behavior}">
                        </div>
                    </div>
                </div>
            `;
        });
        container.innerHTML = html;
    } else {
        if (placeholder) {
            placeholder.src = "assets/noData.png";
            placeholder.style.display = 'block';
        }
    }
}

// =================================================================
// CORE APPLICATION LOGIC
// =================================================================

/**
 * Gathers all checked behaviors and parameters, then calls the backend to generate actograms.
 */
async function generateAndDisplayActograms() {
    const checkedBehaviors = Array.from(document.querySelectorAll('.behavior-checkbox:checked'));
    
    latestVizTaskId++; // Increment to create a new, unique ID for this task.
    const currentTaskId = latestVizTaskId;
    
    if (checkedBehaviors.length === 0) {
        updateActogramDisplay([], currentTaskId);
        document.getElementById('actogram-title').textContent = 'Actogram';
        return;
    }

    const firstCheckbox = checkedBehaviors[0];
    const rootDir = firstCheckbox.dataset.root;
    const sessionDir = firstCheckbox.dataset.session;
    const modelName = firstCheckbox.dataset.model;
    const behaviorNames = checkedBehaviors.map(cb => cb.dataset.behavior);

    const framerate = document.getElementById('vs-framerate').value;
    const binsize = document.getElementById('vs-binsize').value;
    const start = document.getElementById('vs-start').value;
    const threshold = document.getElementById('vs-threshold').value;
    const lightcycle = document.getElementById('vs-lcycle').value;
    const plotAcrophase = document.getElementById('vs-acrophase').checked;

    if (!framerate || !binsize || !start || !threshold) return;

    document.getElementById('actogram-title').textContent = `Actogram: ${modelName} (Recording: ${sessionDir})`;
    showActogramLoadingIndicator();

    try {
        await eel.generate_actograms(
            rootDir, sessionDir, modelName, behaviorNames,
            framerate, binsize, start, threshold, lightcycle, plotAcrophase,
            currentTaskId // Send the unique ID with the request
        )();
    } catch (error) {
        console.error("Error calling eel.generate_actograms:", error);
        updateActogramDisplay([], currentTaskId);
        showErrorOnVisualizePage(`Failed to generate actogram(s): ${error.message || error}`);
    }
}

/**
 * Handles a click on a behavior checkbox. It unchecks behaviors from other
 * models to prevent confusion and then triggers the debounced update.
 */
function handleBehaviorSelection(checkbox) {
    const rootDir = checkbox.dataset.root;
    const sessionDir = checkbox.dataset.session;
    const modelName = checkbox.dataset.model;

    if (currentSelection.root !== rootDir || currentSelection.session !== sessionDir || currentSelection.model !== modelName) {
        document.querySelectorAll('.behavior-checkbox').forEach(cb => {
            if (cb.dataset.model !== modelName || cb.dataset.session !== sessionDir) {
                cb.checked = false;
            }
        });
        currentSelection = { root: rootDir, session: sessionDir, model: modelName };
    }
    
    clearTimeout(actogramDebounceTimer);
    actogramDebounceTimer = setTimeout(generateAndDisplayActograms, 400);
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
            container.innerHTML = "<p class='text-light'>No classified recordings available.</p>";
            showErrorOnVisualizePage('No classified recordings found. Please run inference first.');
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
                                <input class="form-check-input behavior-checkbox" type="checkbox" id="${behaviorId}" 
                                       data-root="${dateStr}" data-session="${sessionName}" data-model="${modelName}" data-behavior="${behaviorName}"
                                       onclick="handleBehaviorSelection(this)">
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
        console.error("Error initializing page:", error);
        container.innerHTML = "<p class='text-danger text-center'>Error loading data.</p>";
        showErrorOnVisualizePage(`Error loading data: ${error.message || error}`);
    }
}


// =================================================================
// PAGE INITIALIZATION & EVENT LISTENERS
// =================================================================

document.addEventListener('DOMContentLoaded', async () => {
    await new Promise(resolve => setTimeout(resolve, 200)); 
    
    initializeVisualizePageContent();

    const adjustmentControlsIds = ['vs-framerate', 'vs-binsize', 'vs-start', 'vs-threshold', 'vs-lcycle', 'vs-acrophase'];
    adjustmentControlsIds.forEach(controlId => {
        const elem = document.getElementById(controlId);
        if (elem) {
            const eventType = (elem.type === 'checkbox' || elem.tagName.toLowerCase() === 'select') ? 'change' : 'input';
            
            elem.addEventListener(eventType, () => {
                clearTimeout(actogramDebounceTimer);
                actogramDebounceTimer = setTimeout(generateAndDisplayActograms, 400);
            });
        }
    });
});

window.addEventListener("unload", () => {
    if (!routingInProgress) { eel.kill_streams()?.().catch(err => console.error(err)); }
});
window.onbeforeunload = () => {
    if (!routingInProgress) { eel.kill_streams()?.().catch(err => console.error(err)); }
};