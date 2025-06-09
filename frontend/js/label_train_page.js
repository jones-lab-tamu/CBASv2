/**
 * @file label_train_page.js
 * Manages the Label & Train page UI and interactions.
 * Handles dataset listing, selection for labeling, video navigation, frame labeling,
 * dataset creation, model training initiation, and inference initiation.
 */

// const ipc = window.ipcRenderer; // Not directly used in this file, but good to keep if future IPC is needed

// --- State Variables ---
// let loaded = false; // General page loaded flag, not heavily used
/** Flag to prevent `kill_streams` from being called during intentional page navigation. */
let routingInProgress = false;
/** True if the labeling interface (video player and controls) is active. */
let labelingInterfaceActive = false;

// Bootstrap Modals
const addDatasetModalElement = document.getElementById('addDataset');
let addDatasetBsModal = addDatasetModalElement ? new bootstrap.Modal(addDatasetModalElement) : null;

const trainModalElement = document.getElementById('trainModal');
let trainBsModal = trainModalElement ? new bootstrap.Modal(trainModalElement) : null;

const inferenceModalElement = document.getElementById('inferenceModal');
let inferenceBsModal = inferenceModalElement ? new bootstrap.Modal(inferenceModalElement) : null;

const errorModalElement = document.getElementById('errorModal'); // General error modal for this page
const errorMessageElement = document.getElementById("error-message");
let generalErrorBsModal = errorModalElement ? new bootstrap.Modal(errorModalElement) : null;


/** 
 * Stores the directory tree for recordings, used in dataset/inference modals.
 * Structure: { "date_dir1": ["session1", "session2"], "date_dir2": [...] }
 * This is populated when showDatasetModal or showInferenceModal is called.
 * @type {Object<string, string[]>} 
 */
let recordingDirTree = {};

/** Navigation speed multiplier for arrow keys during labeling (frames per step). */
let scrubSpeedMultiplier = 1;


// --- Utility Functions ---

/** Shows the error modal with a custom message. */
function showErrorOnLabelTrainPage(message) {
    if (errorMessageElement && generalErrorBsModal) {
        errorMessageElement.innerText = message;
        generalErrorBsModal.show();
    } else {
        alert(message); // Fallback
    }
}

// --- Routing Functions ---
function routeToRecordPage() { routingInProgress = true; window.location.href = './record.html'; }
function routeToLabelTrainPage() { routingInProgress = true; window.location.href = './label-train.html'; }
function routeToVisualizePage() { routingInProgress = true; window.location.href = './visualize.html'; }

// --- Labeling Image Interaction ---

/**
 * Handles mouse move event on the label image timeline area for scrubbing.
 * Calls backend to update frame based on click position.
 * @param {MouseEvent} event - The mouse move event.
 */
function handleMouseMoveForLabelScrub(event) {
    const imageElement = event.target; // Should be the #label-image element
    if (!imageElement) return;
    const imageRect = imageElement.getBoundingClientRect();
    const x = event.clientX - imageRect.left; 
    
    // Assuming the image display width in JS is used for scaling timeline clicks.
    // Using imageRect.width for more dynamic scaling if image size changes.
    // const imageDisplayWidth = imageRect.width || 500; // Fallback to 500 if width is 0
    const imageDisplayWidth = 500; // Sticking to original 500px assumption for consistency with backend

    if (typeof eel !== 'undefined' && eel.handle_click_on_label_image) {
        eel.handle_click_on_label_image(x, 0)(); // y-coordinate is not used by backend for frame seeking here
    }
}

/** Removes mouse move and mouse up listeners for label scrubbing. */
function handleMouseUpForLabelScrub() {
    document.removeEventListener('mousemove', handleMouseMoveForLabelScrub);
    document.removeEventListener('mouseup', handleMouseUpForLabelScrub);
}

const labelImageDisplayElement = document.getElementById('label-image');
if (labelImageDisplayElement) {
    labelImageDisplayElement.addEventListener('mousedown', function (event) {
        const imageRect = event.target.getBoundingClientRect();
        const y = event.clientY - imageRect.top;
        
        // Determine if the click is on the timeline area (e.g., bottom 50px of the image display)
        // This assumes the timeline is rendered as part of the image sent from Python.
        const timelineHeightApproximation = 50; // Approximate height of the timeline area
        if (y > (imageRect.height - timelineHeightApproximation - 10) && y < (imageRect.height -5) ) { // Click is on the timeline part
            const x = event.clientX - imageRect.left;
            if (typeof eel !== 'undefined' && eel.handle_click_on_label_image) {
                eel.handle_click_on_label_image(x, y)(); // Initial click
            }
            document.addEventListener('mousemove', handleMouseMoveForLabelScrub);
            document.addEventListener('mouseup', handleMouseUpForLabelScrub);
        }
    });
}


// --- UI Update Functions (Called by Python via Eel) ---

/** Exposed to Python: Updates the labeling image source. */
eel.expose(updateLabelImageSrc);
function updateLabelImageSrc(base64Val) {
    const elem = document.getElementById('label-image');
    if (elem) {
        elem.src = base64Val ? "data:image/jpeg;base64," + base64Val : "assets/noVideo.png";
    }
}

/** Exposed to Python: Updates the current video file information display. */
eel.expose(updateFileInfo);
function updateFileInfo(filenameStr) {
    const elem = document.getElementById('file-info');
    if (elem) elem.innerText = filenameStr || "No video loaded";
}

/** Exposed to Python: Updates displayed metrics for a dataset on the main page. */
eel.expose(updateMetricsOnPage);
function updateMetricsOnPage(datasetName, behaviorName, metricGroupKey, metricValue) {
    // This maps the Python metric names to the ID suffixes used in the HTML
    const idSuffixMap = {
        'Train #': 'train-count',
        'Test #': 'test-count',
        'Precision': 'precision',
        'Recall': 'recall',
        'F1 Score': 'fscore'
    };
    
    const suffix = idSuffixMap[metricGroupKey];
    if (!suffix) {
        console.warn(`Unknown metric group key: ${metricGroupKey}`);
        return;
    }
    
    const elemId = `${datasetName}-${behaviorName}-${suffix}`;
    const elem = document.getElementById(elemId);
    
    if (elem) {
        // Add a subtle highlight effect to show the value has changed
        elem.innerText = metricValue;
        elem.classList.add('bg-success', 'text-white');
        setTimeout(() => {
            elem.classList.remove('bg-success', 'text-white');
        }, 2000); // Highlight for 2 seconds
    } else {
        console.warn(`UI element with ID not found: ${elemId}`);
    }
}

/** Exposed to Python: Updates behavior instance/frame counts in the labeling UI side panel. */
eel.expose(updateLabelingStats); // Renamed for clarity
function updateLabelingStats(behaviorName, instanceCount, frameCount) {
    const elem = document.getElementById(`controls-${behaviorName}-count`);
    if (elem) {
        elem.innerHTML = `${instanceCount} / ${frameCount}`;
    }
}

/** Exposed to Python: Updates the training status message for a dataset. */
eel.expose(updateTrainingStatusOnUI); // Renamed
function updateTrainingStatusOnUI(datasetName, message) {
    // This function is called by the Python TrainingThread to update the UI.
    // It needs a corresponding HTML element to display the status.
    // Example: <div id="training-status-MyDatasetName"></div> under each dataset card.
    const statusElemId = `dataset-status-${datasetName}`;
    let statusElem = document.getElementById(statusElemId);

    // If the status element doesn't exist yet, we try to create it.
    if (!statusElem) {
        let datasetCard = null;
        // Get all h5 elements that could be card titles.
        const allCardTitles = document.querySelectorAll('.card .card-title');
        
        // Loop through them to find the one with the matching dataset name.
        for (const titleElement of allCardTitles) {
            // Use textContent and trim() to be safe.
            if (titleElement.textContent.trim() === datasetName) {
                // We found the right h5, now find its parent card.
                datasetCard = titleElement.closest('.card');
                break; // Exit the loop once we've found our match.
            }
        }
        
        if (datasetCard) {
            statusElem = document.createElement('div');
            statusElem.id = statusElemId;
            statusElem.className = 'card-footer text-muted small'; // Example styling
            datasetCard.appendChild(statusElem);
        } else {
            // This is the fallback if the card isn't found.
            console.log(`Training status for ${datasetName}: ${message}`); 
            return;
        }
    }
    
    statusElem.innerText = message;
    statusElem.style.display = message ? 'block' : 'none'; // Hide if message is empty
}

eel.expose(updateDatasetLoadProgress);
function updateDatasetLoadProgress(datasetName, percent) {
    const container = document.getElementById(`progress-container-${datasetName}`);
    const bar = document.getElementById(`progress-bar-${datasetName}`);
    
    if (!container || !bar) return;

    if (percent < 0) { // Error signal
        container.style.display = 'none';
        return;
    }
    
    if (percent >= 0 && percent < 100) {
        container.style.display = 'block';
        const displayPercent = Math.round(percent);
        bar.style.width = `${displayPercent}%`;
        bar.innerText = `Loading: ${displayPercent}%`;
    }

    if (percent >= 100) { // Completion signal
        bar.style.width = '100%';
        bar.innerText = 'Loaded!';
        setTimeout(() => {
            container.style.display = 'none';
            bar.style.width = '0%';
            bar.innerText = '';
        }, 1500);
    }
}

// --- Main Page Initialization and Data Loading ---

/** Loads initial dataset configurations and renders them. */
async function loadInitialDatasetCards() {
    try {
        const datasets = await eel.load_dataset_configs()();
        const container = document.getElementById('dataset-container');
        if (!container) { console.error("Dataset container not found"); return; }

        // The container needs the 'row' class for Bootstrap's grid system to work correctly.
        container.className = 'row'; 
        
        let htmlContent = '';
        // Attempt to get project models for JonesLabModel check
        if (await eel.model_exists("JonesLabModel")()) {
             htmlContent += `
                <div class="col-md-6 col-lg-4 mb-3">
                    <div class="card shadow h-100">
                        <div class="card-header bg-dark text-white">
                            <h5 class="card-title mb-0">JonesLabModel <span class="badge bg-info">Default</span></h5>
                        </div>
                        <div class="card-body d-flex flex-column">
                            <p class="card-text small text-muted">Pre-trained model for general inference.</p>
                        </div>
                        <div class="card-footer text-end">
                             <button class="btn btn-sm btn-outline-warning" type="button" onclick="showInferenceModal('JonesLabModel')">Infer</button>
                        </div>
                    </div>
                </div>`;
        }

        if (datasets) {
            for (const datasetName in datasets) {
                if (datasetName === "JonesLabModel" && (await eel.model_exists("JonesLabModel")())) continue;

                const config = datasets[datasetName];
                const behaviors = config.get?.('behaviors', []) || config.behaviors || [];
                const metrics = config.get?.('metrics', {}) || config.metrics || {};
                const modelPath = config.get?.('model', null) || config.model;
                const modelExists = modelPath != null;
                const metricHeaders = ['Train #', 'Test #', 'Precision', 'Recall', 'F1 Score'];

                htmlContent += `
                    <div class="col-md-6 col-lg-4 mb-3">
                        <div class="card shadow h-100">
                            <div class="card-header bg-dark text-white">
                                <h5 class="card-title mb-0">${datasetName}</h5>
                            </div>
                            <div class="card-body" style="font-size: 0.85rem;">`;
                if (behaviors.length > 0) {
                    htmlContent += `<div class="table-responsive"><table class="table table-sm table-hover small"><thead><tr><th>Behavior</th>`;
                    metricHeaders.forEach(h => { htmlContent += `<th class="text-center">${h}</th>`; });
                    htmlContent += `</tr></thead><tbody>`;
                    for (const behaviorName of behaviors) {
                        const bMetrics = metrics[behaviorName] || {};
                        htmlContent += `<tr><td>${behaviorName}</td>`;
                        metricHeaders.forEach((mh, idx) => {
                            let idSuffix = ['train-count', 'test-count', 'precision', 'recall', 'fscore'][idx];
                            htmlContent += `<td class="text-center" id="${datasetName}-${behaviorName}-${idSuffix}">${bMetrics[mh] !== undefined ? bMetrics[mh] : 'N/A'}</td>`;
                        });
                        htmlContent += `</tr>`;
                    }
                    htmlContent += `</tbody></table></div>`;
                } else {
                    htmlContent += `<p class="text-muted">No behaviors defined yet.</p>`;
                }

                // --- START OF ADDED CODE ---
                // This adds the hidden progress bar container to each dataset card.
                // It will be made visible by the 'updateDatasetLoadProgress' function when needed.
                htmlContent += `
                    <div class="progress mt-2" id="progress-container-${datasetName}" style="height: 20px; display: none;">
                        <div class="progress-bar progress-bar-striped progress-bar-animated" id="progress-bar-${datasetName}" role="progressbar" style="width: 0%;" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"></div>
                    </div>`;
                // --- END OF ADDED CODE ---

                htmlContent += `<div id="training-status-${datasetName}" class="mt-2 small text-info"></div>`; // Existing status display area
                
                htmlContent += `</div> <div class="card-footer text-end">
                                <button class="btn btn-sm btn-outline-primary me-1" type="button" onclick="prepareAndShowLabelModal('${datasetName}')">Label</button>
                                <button class="btn btn-sm btn-outline-success me-1" type="button" onclick="showTrainModal('${datasetName}')">Train</button>`;
                if (modelExists) {
                    htmlContent += `<button class="btn btn-sm btn-outline-warning" type="button" onclick="showInferenceModal('${datasetName}')">Infer</button>`;
                }
                htmlContent += `</div></div></div>`;
            }
        }
         if (container) {
            container.innerHTML = htmlContent || "<p class='text-light'>No datasets found. Click '+' to create one.</p>";
         }
    } catch (error) {
        console.error("Error loading initial dataset configs:", error);
        const container = document.getElementById('dataset-container');
        if(container) container.innerHTML = "<p class='text-danger'>Error loading datasets.</p>";
    }
}

// --- Modal Control and Actions ---

/** Submits the training job to the backend. */
async function submitTrainModel() {
    const datasetName = document.getElementById('tm-dataset').innerText;
    const batchSize = document.getElementById('tm-batchsize').value;
    const seqLen = document.getElementById('tm-seqlen').value;
    const learningRate = document.getElementById('tm-lrate').value;
    const epochsCount = document.getElementById('tm-epochs').value;
	const trainMethod = document.getElementById('tm-method').value;

    if (!batchSize || !seqLen || !learningRate || !epochsCount) {
        showErrorOnLabelTrainPage("All training parameters must be filled."); return;
    }
    if (isNaN(parseInt(batchSize)) || isNaN(parseInt(seqLen)) || isNaN(parseFloat(learningRate)) || isNaN(parseInt(epochsCount))) {
        showErrorOnLabelTrainPage("Training parameters must be valid numbers."); return;
    }
    
    try {
        // CHANGED THIS LINE: Replaced eel.updateTrainingStatus(...) with a direct JS call
        updateTrainingStatusOnUI(datasetName, "Training task queued...");
        
        await eel.train_model(datasetName, batchSize, learningRate, epochsCount, seqLen, trainMethod)();
		
        if (trainBsModal) trainBsModal.hide();
        // Python's TrainingThread will call updateTrainingStatusOnUI to update further
    } catch (error) {
        console.error("Error submitting training model task:", error);
        showErrorOnLabelTrainPage(`Failed to submit training task: ${error.message || error}`);
        
        // CHANGED THIS LINE TOO: Fixed the error handler as well
        updateTrainingStatusOnUI(datasetName, "Error queuing task.");
    }
}
/** Populates and shows the training modal for a dataset. */
function showTrainModal(datasetName) {
    const tmDatasetElement = document.getElementById('tm-dataset');
    if (tmDatasetElement) tmDatasetElement.innerText = datasetName;
    if (trainBsModal) trainBsModal.show();
}

/**
 * Determines if text should be black or white based on a background hex color.
 * @param {string} hexColor - The background color in hex format (e.g., "#RRGGBB").
 * @returns {string} '#000000' (black) or '#FFFFFF' (white).
 */
function getTextColorForBg(hexColor) {
    const cleanHex = hexColor.startsWith('#') ? hexColor.slice(1) : hexColor;
    const r = parseInt(cleanHex.substring(0, 2), 16);
    const g = parseInt(cleanHex.substring(2, 4), 16);
    const b = parseInt(cleanHex.substring(4, 6), 16);
    // Standard luminance calculation
    const luminance = (0.299 * r + 0.587 * g + 0.114 * b);
    // Return black for light backgrounds, white for dark backgrounds
    return luminance > 140 ? '#000000' : '#FFFFFF';
}

/** Initializes the labeling UI for the selected dataset. */
async function prepareAndShowLabelModal(datasetName) {
    try {
        const result = await eel.start_labeling(datasetName)();
        if (result && Array.isArray(result) && result.length >= 2) {
            labelingInterfaceActive = true;
            const controlsContainer = document.getElementById('controls');
            if (!controlsContainer) return;

            const behaviors = result[0];
            const colors = result[1]; // These are the hex color strings

            let controlsHTML = `
                <div class="card bg-dark text-light">
                    <div class="card-header">
                        <h5>Behaviors</h5>
                    </div>
                    <ul class="list-group list-group-flush">
            `;

            controlsHTML += `
                <li class="list-group-item bg-dark text-light d-flex justify-content-between">
                    <strong>Behavior</strong>
                    <span><strong>Code</strong></span>
                    <span><strong>Count</strong></span>
                </li>
            `;

            behaviors.forEach((behaviorName, index) => {
                const key = (index < 9) ? (index + 1) : String.fromCharCode('a'.charCodeAt(0) + (index - 9));
                
                // --- START OF CHANGES ---
                const bgColor = colors[index];
                const textColor = getTextColorForBg(bgColor);

                controlsHTML += `
                    <li class="list-group-item bg-dark text-light d-flex justify-content-between align-items-center" 
                        onclick="eel.label_frame(${index})()" style="cursor: pointer;">
                        
                        <span style="flex-basis: 50%;">${behaviorName}</span>
                        
                        <span class="badge rounded-pill" 
                              style="flex-basis: 15%; background-color: ${bgColor}; color: ${textColor};">${key}</span>
                        
                        <span id="controls-${behaviorName}-count" class="badge bg-secondary rounded-pill" style="flex-basis: 25%;">0 / 0</span>
                    </li>
                `;
                // --- END OF CHANGES ---
            });

            controlsHTML += `</ul></div>`;

            controlsContainer.innerHTML = controlsHTML;

            const datasetsView = document.getElementById('datasets');
            const labelView = document.getElementById('label');
            if (datasetsView) datasetsView.style.display = 'none';
            if (labelView) labelView.style.display = 'flex';
            
            document.getElementById('labeling-cheat-sheet').style.display = 'block';
            
            await eel.update_counts()();
        } else {
            showErrorOnLabelTrainPage('Error starting labeling: No videos found or invalid dataset.');
        }
    } catch (error) {
        console.error("Error in prepareAndShowLabelModal:", error);
        const errorMsg = error.errorText ? error.errorText : (error.message || 'Unknown error');
        showErrorOnLabelTrainPage(`Error initializing labeling: ${errorMsg}`);
    }
}


/** Creates a new dataset via the backend. */
async function submitCreateDataset() { // Renamed for clarity
    const selectedRecordings = [];
    const dateDirs = Object.keys(recordingDirTree); // Use the populated tree

    for (const dir of dateDirs) {
        const dirCheckbox = document.getElementById(dir); // Assuming IDs in modal are just dir names
        if (dirCheckbox && dirCheckbox.checked) {
            selectedRecordings.push(dir);
        } else {
            const subdirs = recordingDirTree[dir] || [];
            for (const subdir of subdirs) {
                const subdirCheckbox = document.getElementById(`${dir}-${subdir}`);
                if (subdirCheckbox && subdirCheckbox.checked) {
                    selectedRecordings.push(`${dir}/${subdir}`); // Use / for consistency
                }
            }
        }
    }
    
    if (selectedRecordings.length === 0) {
         if (!confirm("No recordings selected. Create dataset with empty whitelist?")) return;
    }

    const name = document.getElementById('dataset-name-modal-input').value; // Assuming specific modal input IDs
    const behaviorsStr = document.getElementById('dataset-behaviors-modal-input').value;

    if (!name.trim() || !behaviorsStr.trim()) { showErrorOnLabelTrainPage("Name and behaviors required."); return; }
    const behavior_array = behaviorsStr.split(';').map(b => b.trim()).filter(b => b);
    if (behavior_array.length < 1 || behavior_array.length > 20) { showErrorOnLabelTrainPage('1-20 behaviors required.'); return; }

    try {
        const success = await eel.create_dataset(name, behavior_array, selectedRecordings)();
        if (success) {
            if (addDatasetBsModal) addDatasetBsModal.hide();
            document.getElementById('dataset-name-modal-input').value = "";
            document.getElementById('dataset-behaviors-modal-input').value = "";
            const treeElem = document.getElementById('ad-recording-tree');
            if(treeElem) treeElem.innerHTML = ""; 
            recordingDirTree = {};
            await loadInitialDatasetCards();
        } else { showErrorOnLabelTrainPage('Failed to create dataset (already exists or error).'); }
    } catch (e) { console.error("Create dataset error:", e); showErrorOnLabelTrainPage(`Error: ${e.message||e}`); }
}

/** Handles page unload events. */
window.addEventListener("unload", function () {
    if (!routingInProgress) { if (typeof eel !== 'undefined' && eel.kill_streams) eel.kill_streams()().catch(err => console.error("Unload kill_streams error:", err));}
});
window.onbeforeunload = function () {
    if (!routingInProgress) { if (typeof eel !== 'undefined' && eel.kill_streams) eel.kill_streams()().catch(err => console.error("Beforeunload kill_streams error:", err));}
    // return null;
};

/** Updates child checkboxes when a parent directory checkbox is changed. */
function updateChildrenCheckboxes(parentCheckboxId, isInfModalSuffix = false) {
    const baseParentDirId = isInfModalSuffix ? parentCheckboxId.slice(0, -3) : parentCheckboxId;
    const subdirs = recordingDirTree[baseParentDirId];
    const parentCheckbox = document.getElementById(parentCheckboxId);

    if (subdirs && parentCheckbox) {
        subdirs.forEach(subdir => {
            const childCheckboxId = isInfModalSuffix ? `${baseParentDirId}-${subdir}-im` : `${baseParentDirId}-${subdir}`;
            const childCheckbox = document.getElementById(childCheckboxId);
            if (childCheckbox) childCheckbox.checked = parentCheckbox.checked;
            // else console.warn(`Child checkbox ${childCheckboxId} not found for parent ${parentCheckboxId}`);
        });
    }
}

/** Populates and shows the "Add Dataset" modal. */
async function showAddDatasetModal() { // Renamed
    try {
        const fetchedRecordingTree = await eel.get_record_tree()();
        const treeContainer = document.getElementById('ad-recording-tree');
        if (!treeContainer) {console.error("Add dataset tree container not found"); return;}
        treeContainer.innerHTML = ''; 
        recordingDirTree = fetchedRecordingTree || {}; // Update global tree

        if (fetchedRecordingTree && Object.keys(fetchedRecordingTree).length > 0) {
            for (const dateDir in fetchedRecordingTree) {
                const sessionDirs = fetchedRecordingTree[dateDir];
                let dateHTML = `<div class="form-check"><input class="form-check-input" type="checkbox" id="${dateDir}" onchange="updateChildrenCheckboxes('${dateDir}', false)"><label class="form-check-label" for="${dateDir}">${dateDir}</label></div>`;
                let sessionsHTML = "";
                if (sessionDirs.length > 0) {
                    sessionsHTML += `<div style='margin-left:20px'>`;
                    sessionDirs.forEach(sessionDir => {
                        sessionsHTML += `<div class="form-check"><input class="form-check-input" type="checkbox" id="${dateDir}-${sessionDir}"><label class="form-check-label" for="${dateDir}-${sessionDir}">${sessionDir}</label></div>`;
                    });
                    sessionsHTML += `</div>`;
                }
                treeContainer.innerHTML += dateHTML + sessionsHTML;
            }
            if (addDatasetBsModal) addDatasetBsModal.show();
        } else { showErrorOnLabelTrainPage('No recordings found to create a dataset from.'); }
    } catch (e) { console.error("Show dataset modal error:", e); showErrorOnLabelTrainPage("Failed to load recording tree."); }
}

/** Populates and shows the inference modal. */
async function showInferenceModal(datasetName) {
    const imDatasetElem = document.getElementById('im-dataset');
    if(imDatasetElem) imDatasetElem.innerText = datasetName;
    const treeContainer = document.getElementById('im-recording-tree');
    if (!treeContainer) {console.error("Inference tree container not found"); return;}
    treeContainer.innerHTML = ''; 
    recordingDirTree = {}; // Reset/repopulate for this modal's context

    try {
        const fetchedRecordingTree = await eel.get_record_tree()();
        if (fetchedRecordingTree && Object.keys(fetchedRecordingTree).length > 0) {
            recordingDirTree = fetchedRecordingTree; // Populate for this modal
            for (const dateDir in fetchedRecordingTree) {
                const sessionDirs = fetchedRecordingTree[dateDir];
                let dateHTML = `<div class="form-check"><input class="form-check-input" type="checkbox" id="${dateDir}-im" onchange="updateChildrenCheckboxes('${dateDir}-im', true)"><label class="form-check-label" for="${dateDir}-im">${dateDir}</label></div>`;
                let sessionsHTML = "";
                if (sessionDirs.length > 0) {
                    sessionsHTML += `<div style='margin-left:20px'>`;
                    sessionDirs.forEach(sessionDir => {
                        sessionsHTML += `<div class="form-check"><input class="form-check-input" type="checkbox" id="${dateDir}-${sessionDir}-im"><label class="form-check-label" for="${dateDir}-${sessionDir}-im">${sessionDir}</label></div>`;
                    });
                    sessionsHTML += `</div>`;
                }
                treeContainer.innerHTML += dateHTML + sessionsHTML;
            }
            if (inferenceBsModal) inferenceBsModal.show();
        } else { showErrorOnLabelTrainPage('No recordings found to run inference on.'); }
    } catch (e) { console.error("Show inference modal error:", e); showErrorOnLabelTrainPage("Failed to load recording tree."); }
}

/** Starts classification (inference). */
async function submitStartClassification() { // Renamed
    const datasetNameForModel = document.getElementById('im-dataset').innerText;
    const selectedRecs = [];
    const dateDirs = Object.keys(recordingDirTree);

    for (const dir of dateDirs) {
        const dirCheckbox = document.getElementById(`${dir}-im`);
        if (dirCheckbox && dirCheckbox.checked) { selectedRecs.push(dir); }
        else {
            const subdirs = recordingDirTree[dir] || [];
            for (const subdir of subdirs) {
                const subdirCheckbox = document.getElementById(`${dir}-${subdir}-im`);
                if (subdirCheckbox && subdirCheckbox.checked) { selectedRecs.push(`${dir}/${subdir}`); }
            }
        }
    }
    if (selectedRecs.length === 0) { showErrorOnLabelTrainPage('No recordings selected for inference.'); return; }

    const loadingSpinner = document.getElementById('loading-spinner-general'); // Use a general spinner
    if (loadingSpinner) loadingSpinner.style.display = 'block'; // Show
    
    try {
        await eel.start_classification(datasetNameForModel, selectedRecs)();
        updateTrainingStatusOnUI(datasetNameForModel, "Inference tasks queued..."); // Use the same status update
    } catch (e) { console.error("Start classification error:", e); showErrorOnLabelTrainPage(`Failed to start: ${e.message||e}`); }
    finally { 
        if (loadingSpinner) loadingSpinner.style.display = 'none'; // Hide
        if (inferenceBsModal) inferenceBsModal.hide(); 
    }
}

// Keyboard Event Listeners for Labeling
window.addEventListener("keydown", (event) => {
    if (labelingInterfaceActive && document.getElementById('label')?.style.display === 'flex') {
        let handled = true;
        switch (event.key) {
            case "ArrowLeft": (event.ctrlKey||event.metaKey) ? eel.next_video(-1)() : eel.next_frame(-scrubSpeedMultiplier)(); break;
            case "ArrowRight": (event.ctrlKey||event.metaKey) ? eel.next_video(1)() : eel.next_frame(scrubSpeedMultiplier)(); break;
            case "ArrowUp": scrubSpeedMultiplier = Math.min(scrubSpeedMultiplier * 2, 128); console.log("Scrub speed:", scrubSpeedMultiplier); break;
            case "ArrowDown": scrubSpeedMultiplier = Math.max(1, Math.trunc(scrubSpeedMultiplier / 2)); console.log("Scrub speed:", scrubSpeedMultiplier); break;
            case "Delete": eel.delete_instance()(); break;
            case "Backspace": eel.pop_instance()(); break;
            default:
                let bIdx = -1;
                if (event.keyCode >= 49 && event.keyCode <= 57) bIdx = event.keyCode - 49; // 1-9
                else if (event.keyCode >= 65 && event.keyCode <= 90) bIdx = event.keyCode - 65 + 9; // a-z
                if (bIdx !== -1) eel.label_frame(bIdx)(); else handled = false;
                break;
        }
        if (handled) event.preventDefault();
    }
});

// This function will wait until the Eel WebSocket is fully connected.
function waitForEelConnection() {
    return new Promise(resolve => {
        // If connection is already open, resolve immediately.
        if (eel._websocket && eel._websocket.readyState === 1) {
            resolve();
            return;
        }
        // Otherwise, wait for the 'open' event.
        // We can check periodically as a fallback.
        const interval = setInterval(() => {
            if (eel._websocket && eel._websocket.readyState === 1) {
                clearInterval(interval);
                resolve();
            }
        }, 100); // Check every 100ms
    });
}


// Initial load
document.addEventListener('DOMContentLoaded', async () => {
    // Wait for the connection to be ready before doing anything else.
    await waitForEelConnection();
    
    // Now that we are sure the connection is open, load the data.
    loadInitialDatasetCards();

    // Assign submit buttons for modals
    const createDatasetBtn = document.getElementById('createDatasetButton');
    if(createDatasetBtn) createDatasetBtn.onclick = submitCreateDataset;
    const trainModelBtn = document.getElementById('trainModelButton');
    if(trainModelBtn) trainModelBtn.onclick = submitTrainModel;
    const startClassificationBtn = document.getElementById('startClassificationButton');
    if(startClassificationBtn) startClassificationBtn.onclick = submitStartClassification;
});