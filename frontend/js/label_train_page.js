/**
 * @file Manages the Label & Train page UI and interactions.
 * @description This file handles dataset listing, model training, inference, and the entire
 * advanced labeling workflow including pre-labeling and instance-based navigation.
 */

// =================================================================
// GLOBAL STATE & VARIABLES
// =================================================================

/** Flag to prevent `kill_streams` from being called during intentional page navigation. */
let routingInProgress = false;
/** True if the labeling interface is active, enabling keyboard shortcuts. */
let labelingInterfaceActive = false;
/** Navigation speed multiplier for arrow keys during labeling. */
let scrubSpeedMultiplier = 1;

/**
 * Stores the directory tree for recordings, populated when a modal needs it.
 * @type {Object<string, string[]>}
 */
let recordingDirTree = {};

// --- Bootstrap Modal Instances ---
const addDatasetModalElement = document.getElementById('addDataset');
const trainModalElement = document.getElementById('trainModal');
const inferenceModalElement = document.getElementById('inferenceModal');
const errorModalElement = document.getElementById('errorModal');
const preLabelModalElement = document.getElementById('preLabelModal');

let addDatasetBsModal = addDatasetModalElement ? new bootstrap.Modal(addDatasetModalElement) : null;
let trainBsModal = trainModalElement ? new bootstrap.Modal(trainModalElement) : null;
let inferenceBsModal = inferenceModalElement ? new bootstrap.Modal(inferenceModalElement) : null;
let generalErrorBsModal = errorModalElement ? new bootstrap.Modal(errorModalElement) : null;
let preLabelBsModal = preLabelModalElement ? new bootstrap.Modal(preLabelModalElement) : null;


// =================================================================
// ROUTING & UTILITY FUNCTIONS
// =================================================================

function routeToRecordPage() { routingInProgress = true; window.location.href = './record.html'; }
function routeToLabelTrainPage() { routingInProgress = true; window.location.href = './label-train.html'; }
function routeToVisualizePage() { routingInProgress = true; window.location.href = './visualize.html'; }

/**
 * Determines if text should be black or white for good contrast against a background color.
 * @param {string} hexColor - The background color in hex format (e.g., "#RRGGBB").
 * @returns {string} '#000000' (black) or '#FFFFFF' (white).
 */
function getTextColorForBg(hexColor) {
    if (!hexColor) return '#000000';
    const cleanHex = hexColor.startsWith('#') ? hexColor.slice(1) : hexColor;
    if (cleanHex.length !== 6) return '#000000';
    const r = parseInt(cleanHex.substring(0, 2), 16);
    const g = parseInt(cleanHex.substring(2, 4), 16);
    const b = parseInt(cleanHex.substring(4, 6), 16);
    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    return luminance > 0.5 ? '#000000' : '#FFFFFF';
}


// =================================================================
// EEL-EXPOSED FUNCTIONS (Called FROM Python)
// =================================================================

/** Shows the general error modal with a custom message. */
eel.expose(showErrorOnLabelTrainPage);
function showErrorOnLabelTrainPage(message) {
    const errorMessageElement = document.getElementById("error-message");
    if (errorMessageElement && generalErrorBsModal) {
        errorMessageElement.innerText = message;
        generalErrorBsModal.show();
    } else {
        alert(message); // Fallback
    }
}

/** Updates the labeling image source with a new base64 encoded frame. */
eel.expose(updateLabelImageSrc);
function updateLabelImageSrc(base64Val) {
    const elem = document.getElementById('label-image');
    if (elem) {
        elem.src = base64Val ? "data:image/jpeg;base64," + base64Val : "assets/noVideo.png";
    }
}

/** Updates the file path information at the top of the labeling UI. */
eel.expose(updateFileInfo);
function updateFileInfo(filenameStr) {
    const elem = document.getElementById('file-info');
    if (elem) elem.innerText = filenameStr || "No video loaded";
}

/** Updates behavior instance/frame counts in the labeling UI side panel. */
eel.expose(updateLabelingStats);
function updateLabelingStats(behaviorName, instanceCount, frameCount) {
    const elem = document.getElementById(`controls-${behaviorName}-count`);
    if (elem) {
        elem.innerHTML = `${instanceCount} / ${frameCount}`;
    }
}

/** Updates displayed metrics for a dataset on the main dataset list page. */
eel.expose(updateMetricsOnPage);
function updateMetricsOnPage(datasetName, behaviorName, metricGroupKey, metricValue) {
    const idSuffixMap = {
        'Train #': 'train-count',
        'Test #': 'test-count',
        'Precision': 'precision',
        'Recall': 'recall',
        'F1 Score': 'fscore'
    };
    const suffix = idSuffixMap[metricGroupKey];
    if (!suffix) return;
    const elem = document.getElementById(`${datasetName}-${behaviorName}-${suffix}`);
    if (elem) {
        elem.innerText = metricValue;
        elem.classList.add('bg-success', 'text-white');
        setTimeout(() => {
            elem.classList.remove('bg-success', 'text-white');
        }, 2000);
    }
}

/** Updates the training status message for a dataset card. */
eel.expose(updateTrainingStatusOnUI);
function updateTrainingStatusOnUI(datasetName, message) {
    const statusElem = document.getElementById(`dataset-status-${datasetName}`);
    if (statusElem) {
        statusElem.innerText = message;
        statusElem.style.display = message ? 'block' : 'none';
    }
}

/** Updates the dataset loading progress bar during training prep. */
eel.expose(updateDatasetLoadProgress);
function updateDatasetLoadProgress(datasetName, percent) {
    const container = document.getElementById(`progress-container-${datasetName}`);
    const bar = document.getElementById(`progress-bar-${datasetName}`);
    if (!container || !bar) return;

    if (percent < 0) { // Error signal
        container.style.display = 'none';
    } else if (percent >= 100) { // Completion signal
        bar.style.width = '100%';
        bar.innerText = 'Loaded!';
        setTimeout(() => {
            container.style.display = 'none';
            bar.style.width = '0%';
            bar.innerText = '';
        }, 1500);
    } else { // In-progress
        container.style.display = 'block';
        const displayPercent = Math.round(percent);
        bar.style.width = `${displayPercent}%`;
        bar.innerText = `Loading: ${displayPercent}%`;
    }
}

/** Dynamically builds the main behavior panel in the labeling UI. */
eel.expose(buildLabelingUI);
function buildLabelingUI(behaviors, colors) {
    labelingInterfaceActive = true;
    const controlsContainer = document.getElementById('controls');
    if (!controlsContainer) return;

    let controlsHTML = `<div class="card bg-dark text-light h-100"><div class="card-header"><h5></h5></div><ul class="list-group list-group-flush">`;
    controlsHTML += `<li class="list-group-item bg-dark text-light d-flex justify-content-between"><strong>Behavior</strong><span><strong>Key</strong></span><span><strong>Count</strong></span></li>`;

    if (behaviors && colors && behaviors.length === colors.length) {
        behaviors.forEach((behaviorName, index) => {
            const key = (index < 9) ? (index + 1) : String.fromCharCode('a'.charCodeAt(0) + (index - 9));
            const bgColor = colors[index];
            const textColor = getTextColorForBg(bgColor);
            controlsHTML += `
                <li class="list-group-item bg-dark text-light d-flex justify-content-between align-items-center" 
                    onclick="eel.label_frame(${index})()" style="cursor: pointer;" title="Click or press '${key}' to label '${behaviorName}'">
                    <span style="flex-basis: 50%;">${behaviorName}</span>
                    <span class="badge rounded-pill" style="flex-basis: 15%; background-color: ${bgColor}; color: ${textColor};">${key}</span>
                    <span id="controls-${behaviorName}-count" class="badge bg-secondary rounded-pill" style="flex-basis: 25%;">0 / 0</span>
                </li>`;
        });
    }
    controlsHTML += `</ul></div>`;
    controlsContainer.innerHTML = controlsHTML;

    document.getElementById('datasets').style.display = 'none';
    document.getElementById('label').style.display = 'flex';
    document.getElementById('labeling-cheat-sheet').style.display = 'block';
}

/** Sets the UI mode for labeling (scratch vs. review) and updates controls text. */
eel.expose(setLabelingModeUI);
function setLabelingModeUI(mode, modelName = '') {
    const controlsHeader = document.querySelector('#controls .card-header');
    const cheatSheet = document.getElementById('labeling-cheat-sheet');
    if (!controlsHeader || !cheatSheet) return;

    if (mode === 'review') {
        controlsHeader.classList.remove('bg-dark');
        controlsHeader.classList.add('bg-success');
        controlsHeader.querySelector('h5').innerHTML = `Reviewing: <span class="badge bg-light text-dark">${modelName}</span>`;
        cheatSheet.innerHTML = `
            <div class="card bg-dark">
              <div class="card-header"><h5 class="text-success"><i class="bi bi-robot me-2"></i>Review Mode Controls</h5></div>
              <div class="card-body" style="font-size: 0.9rem;">
                <div class="row">
                  <div class="col-md-6">
                    <ul class="list-unstyled">
                      <li><kbd>Tab</kbd> : Jump to Next Instance</li>
                      <li><kbd>Shift</kbd> + <kbd>Tab</kbd> : Jump to Previous Instance</li>
                      <li><kbd>←</kbd> / <kbd>→</kbd> : Step one frame</li>
                      <li><kbd>Click Timeline</kbd> : Jump to frame</li>
                    </ul>
                  </div>
                  <div class="col-md-6">
                    <ul class="list-unstyled">
                      <li><kbd>1</kbd> - <kbd>9</kbd> : Change Instance Label / Start New Label</li>
                      <li><kbd>Delete</kbd> : Delete instance at current frame</li>
                      <li><kbd>Backspace</kbd> : Undo last added label</li>
                      <li><kbd>Ctrl</kbd> + <kbd>←</kbd> / <kbd>→</kbd> : Previous / Next video</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>`;
    } else { // 'scratch' mode
        controlsHeader.classList.remove('bg-success');
        controlsHeader.classList.add('bg-dark');
        controlsHeader.querySelector('h5').innerHTML = `Behaviors (Click to label)`;
        cheatSheet.innerHTML = `
            <div class="card bg-dark">
              <div class="card-header"><h5>Labeling Controls</h5></div>
              <div class="card-body" style="font-size: 0.9rem;">
                <div class="row">
                  <div class="col-md-6">
                    <ul class="list-unstyled">
                      <li><kbd>←</kbd> / <kbd>→</kbd> : Step one frame</li>
                      <li><kbd>↑</kbd> / <kbd>↓</kbd> : Double / Halve scrub speed</li>
                      <li><kbd>Click Timeline</kbd> : Jump to frame</li>
                    </ul>
                  </div>
                  <div class="col-md-6">
                    <ul class="list-unstyled">
                      <li><kbd>1</kbd> - <kbd>9</kbd> : Start / End a new label</li>
                      <li><kbd>Delete</kbd> : Delete instance at current frame</li>
                      <li><kbd>Backspace</kbd> : Undo last added label</li>
                      <li><kbd>Ctrl</kbd> + <kbd>←</kbd> / <kbd>→</kbd> : Previous / Next video</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>`;
    }
}

/** Highlights the specified row in the behavior panel. */
eel.expose(highlightBehaviorRow);
function highlightBehaviorRow(behaviorNameToHighlight) {
    document.querySelectorAll('#controls .list-group-item').forEach(row => {
        row.classList.remove('highlight-selected');
    });
    if (behaviorNameToHighlight) {
        const behaviorSpans = document.querySelectorAll('#controls .list-group-item span:first-child');
        const targetSpan = Array.from(behaviorSpans).find(span => span.textContent === behaviorNameToHighlight);
        if (targetSpan) {
            targetSpan.closest('.list-group-item')?.classList.add('highlight-selected');
        }
    }
}


// =================================================================
// UI INTERACTION & EVENT HANDLERS (Called FROM HTML)
// =================================================================

/** Saves all corrections for the current video session to file. */
function saveCorrections() {
    if (confirm("Are you sure you want to finalize and save all labels for this video? This will overwrite any previous labels for this file.")) {
        eel.save_session_labels()();
        alert("Labels for this video have been saved!");
    }
}

/** Jumps to the next or previous labeled instance in the video. */
function jumpToInstance(direction) {
    eel.jump_to_instance(direction)();
}

/** Opens the 'Labeling Options' modal and populates its fields. */
async function showPreLabelOptions(datasetName) {
    if (!preLabelBsModal) return;
    document.getElementById('pl-dataset-name').innerText = datasetName;
    const modelSelect = document.getElementById('pl-model-select');
    const sessionSelect = document.getElementById('pl-session-select');
    const videoSelect = document.getElementById('pl-video-select');
    const mainButton = document.querySelector('#preLabelModal .btn-outline-success');
    const scratchButton = document.querySelector('#preLabelModal .btn-outline-primary');

    // Reset UI to default state for "Pre-Label"
    document.querySelector('label[for="pl-model-select"]').style.display = 'block';
    document.querySelector('label[for="pl-session-select"]').style.display = 'block';
    modelSelect.style.display = 'block';
    sessionSelect.style.display = 'block';
    mainButton.innerHTML = '<i class="bi bi-robot me-2"></i>Pre-Label & Correct Selected Video';
    mainButton.onclick = startPreLabeling;
    scratchButton.style.display = 'block';

    sessionSelect.innerHTML = '<option selected disabled>First, choose a model...</option>';
    videoSelect.innerHTML = '<option selected disabled>First, choose a session...</option>';
    sessionSelect.disabled = true;
    videoSelect.disabled = true;

    // Populate model dropdown
    modelSelect.innerHTML = '<option selected disabled>Choose a model...</option>';
    const models = await eel.get_available_models()();
    if (models) models.forEach(modelName => modelSelect.innerHTML += `<option value="${modelName}">${modelName}</option>`);

    // Add event listeners for chained dropdowns
    modelSelect.onchange = async function() {
        sessionSelect.disabled = false;
        videoSelect.disabled = true; videoSelect.innerHTML = '<option>First, choose a session...</option>';
        sessionSelect.innerHTML = '<option>Loading sessions...</option>';
        const sessions = await eel.get_inferred_session_dirs(datasetName, this.value)();
        sessionSelect.innerHTML = '<option selected disabled>Choose a session...</option>';
        if (sessions && sessions.length > 0) {
            sessions.forEach(dir => sessionSelect.innerHTML += `<option value="${dir}">${dir}</option>`);
        } else {
            sessionSelect.innerHTML = '<option>No inferred sessions found</option>';
        }
    };
    sessionSelect.onchange = async function() {
        videoSelect.disabled = false;
        videoSelect.innerHTML = '<option>Loading videos...</option>';
        const videos = await eel.get_inferred_videos_for_session(this.value, modelSelect.value)();
        videoSelect.innerHTML = '<option selected disabled>Choose a video...</option>';
        if (videos && videos.length > 0) {
            videos.forEach(v => videoSelect.innerHTML += `<option value="${v[0]}">${v[1]}</option>`);
        } else {
            videoSelect.innerHTML = '<option>No videos found</option>';
        }
    };
    preLabelBsModal.show();
}

/** Modifies the 'Labeling Options' modal for the "Start From Scratch" workflow. */
async function showVideoSelectionForScratch() {
    const datasetName = document.getElementById('pl-dataset-name').innerText;
    const modelLabel = document.querySelector('label[for="pl-model-select"]');
    const modelSelect = document.getElementById('pl-model-select');
    const sessionLabel = document.querySelector('label[for="pl-session-select"]');
    const sessionSelect = document.getElementById('pl-session-select');
    const mainButton = document.querySelector('#preLabelModal .btn-outline-success');
    const scratchButton = document.querySelector('#preLabelModal .btn-outline-primary');
    const videoSelect = document.getElementById('pl-video-select');
    const prelabelHeader = document.querySelector('#preLabelModal p.text-center');

    // Reconfigure UI for scratch workflow
    if (modelLabel) modelLabel.style.display = 'none';
    if (modelSelect) modelSelect.style.display = 'none';
    if (sessionLabel) sessionLabel.style.display = 'none';
    if (sessionSelect) sessionSelect.style.display = 'none';
    if (scratchButton) scratchButton.style.display = 'none';
    if (prelabelHeader) prelabelHeader.style.display = 'none';

    if (mainButton) {
        mainButton.innerHTML = '<i class="bi bi-pen me-2"></i>Label Selected Video';
        mainButton.onclick = function() {
            const videoPath = videoSelect.value;
            if (!videoPath || videoPath.includes("...")) {
                showErrorOnLabelTrainPage("Please select a video to label.");
                return;
            }
            if (preLabelBsModal) preLabelBsModal.hide();
            prepareAndShowLabelModal(datasetName, videoPath);
        };
    }
    
    // Populate video dropdown
    videoSelect.disabled = false;
    videoSelect.innerHTML = '<option>Loading videos...</option>';
    const videos = await eel.get_videos_for_dataset(datasetName)();
    videoSelect.innerHTML = '<option selected disabled>Choose a video...</option>';
    if (videos && videos.length > 0) {
        videos.forEach(v => videoSelect.innerHTML += `<option value="${v[0]}">${v[1]}</option>`);
    } else {
        videoSelect.innerHTML = '<option>No videos found in dataset</option>';
    }
}

/** Kicks off the pre-labeling workflow. */
async function startPreLabeling() {
    const datasetName = document.getElementById('pl-dataset-name').innerText;
    const modelName = document.getElementById('pl-model-select').value;
    const videoPath = document.getElementById('pl-video-select').value;

    if (!modelName || modelName.includes("...")) { showErrorOnLabelTrainPage("Please select a model."); return; }
    if (!videoPath || videoPath.includes("...")) { showErrorOnLabelTrainPage("Please select a video."); return; }

    if (preLabelBsModal) preLabelBsModal.hide();
    const loadingSpinner = document.getElementById('loading-spinner-general'); // Needs to exist in HTML
    if (loadingSpinner) loadingSpinner.style.display = 'block';

    try {
        const success = await eel.start_labeling_with_preload(datasetName, modelName, videoPath)();
        if (!success) {
            showErrorOnLabelTrainPage("Pre-labeling failed. The backend task could not be started.");
        }
    } catch (e) {
        showErrorOnLabelTrainPage(`An error occurred: ${e.message || e}`);
    } finally {
        if (loadingSpinner) loadingSpinner.style.display = 'none';
    }
}

/** Initiates the labeling UI for a specific video, starting from scratch. */
async function prepareAndShowLabelModal(datasetName, videoToOpen) {
    try {
        const success = await eel.start_labeling(datasetName, videoToOpen, null)();
        if (!success) {
            showErrorOnLabelTrainPage('Backend failed to start the labeling task.');
        }
    } catch (error) {
        showErrorOnLabelTrainPage(`Error initializing labeling interface: ${error.message || 'Unknown error'}`);
    }
}


// =================================================================
// DATASET & MODEL MANAGEMENT FUNCTIONS
// =================================================================

/** Populates and shows the "Add Dataset" modal. */
async function showAddDatasetModal() {
    try {
        const treeContainer = document.getElementById('ad-recording-tree');
        treeContainer.innerHTML = '';
        const fetchedRecordingTree = await eel.get_record_tree()();
        recordingDirTree = fetchedRecordingTree || {};
        if (fetchedRecordingTree && Object.keys(fetchedRecordingTree).length > 0) {
            for (const dateDir in fetchedRecordingTree) {
                let dateHTML = `<div class="form-check"><input class="form-check-input" type="checkbox" id="${dateDir}" onchange="updateChildrenCheckboxes('${dateDir}')"><label class="form-check-label" for="${dateDir}">${dateDir}</label></div>`;
                let sessionsHTML = "<div style='margin-left:20px'>";
                fetchedRecordingTree[dateDir].forEach(sessionDir => {
                    sessionsHTML += `<div class="form-check"><input class="form-check-input" type="checkbox" id="${dateDir}-${sessionDir}"><label class="form-check-label" for="${dateDir}-${sessionDir}">${sessionDir}</label></div>`;
                });
                sessionsHTML += `</div>`;
                treeContainer.innerHTML += dateHTML + sessionsHTML;
            }
            addDatasetBsModal?.show();
        } else {
            showErrorOnLabelTrainPage('No recordings found to create a dataset from.');
        }
    } catch (e) {
        showErrorOnLabelTrainPage("Failed to load recording tree.");
    }
}

/** Submits the form to create a new dataset. */
async function submitCreateDataset() {
    const selectedRecordings = [];
    Object.keys(recordingDirTree).forEach(dir => {
        const dirCheckbox = document.getElementById(dir);
        if (dirCheckbox?.checked) {
            selectedRecordings.push(dir);
        } else {
            recordingDirTree[dir]?.forEach(subdir => {
                const subdirCheckbox = document.getElementById(`${dir}-${subdir}`);
                if (subdirCheckbox?.checked) {
                    selectedRecordings.push(`${dir}/${subdir}`);
                }
            });
        }
    });

    if (selectedRecordings.length === 0 && !confirm("No recordings selected. Create dataset with empty whitelist?")) return;

    const name = document.getElementById('dataset-name-modal-input').value;
    const behaviorsStr = document.getElementById('dataset-behaviors-modal-input').value;
    if (!name.trim() || !behaviorsStr.trim()) { showErrorOnLabelTrainPage("Name and behaviors are required."); return; }
    const behavior_array = behaviorsStr.split(';').map(b => b.trim()).filter(Boolean);
    if (behavior_array.length < 1 || behavior_array.length > 20) { showErrorOnLabelTrainPage('Must have between 1 and 20 behaviors.'); return; }

    const success = await eel.create_dataset(name, behavior_array, selectedRecordings)();
    if (success) {
        addDatasetBsModal?.hide();
        document.getElementById('dataset-name-modal-input').value = '';
        document.getElementById('dataset-behaviors-modal-input').value = '';
        loadInitialDatasetCards();
    } else {
        showErrorOnLabelTrainPage('Failed to create dataset. A dataset with that name may already exist.');
    }
}

/** Populates and shows the training modal. */
function showTrainModal(datasetName) {
    const tmDatasetElement = document.getElementById('tm-dataset');
    if (tmDatasetElement) tmDatasetElement.innerText = datasetName;
    trainBsModal?.show();
}

/** Submits the training job to the backend. */
async function submitTrainModel() {
    const datasetName = document.getElementById('tm-dataset').innerText;
    const batchSize = document.getElementById('tm-batchsize').value;
    const seqLen = document.getElementById('tm-seqlen').value;
    const learningRate = document.getElementById('tm-lrate').value;
    const epochsCount = document.getElementById('tm-epochs').value;
    const trainMethod = document.getElementById('tm-method').value;

    if (!batchSize || !seqLen || !learningRate || !epochsCount) { showErrorOnLabelTrainPage("All training parameters must be filled."); return; }
    
    updateTrainingStatusOnUI(datasetName, "Training task queued...");
    await eel.train_model(datasetName, batchSize, learningRate, epochsCount, seqLen, trainMethod)();
    trainBsModal?.hide();
}

/** Populates and shows the inference modal. */
async function showInferenceModal(datasetName) {
    const imDatasetElem = document.getElementById('im-dataset');
    if (imDatasetElem) imDatasetElem.innerText = datasetName;
    
    const treeContainer = document.getElementById('im-recording-tree');
    treeContainer.innerHTML = '';
    const fetchedRecordingTree = await eel.get_record_tree()();
    recordingDirTree = fetchedRecordingTree || {};
    
    if (fetchedRecordingTree && Object.keys(fetchedRecordingTree).length > 0) {
        for (const dateDir in fetchedRecordingTree) {
            let dateHTML = `<div class="form-check"><input class="form-check-input" type="checkbox" id="${dateDir}-im" onchange="updateChildrenCheckboxes('${dateDir}-im', true)"><label class="form-check-label" for="${dateDir}-im">${dateDir}</label></div>`;
            let sessionsHTML = "<div style='margin-left:20px'>";
            fetchedRecordingTree[dateDir].forEach(sessionDir => {
                sessionsHTML += `<div class="form-check"><input class="form-check-input" type="checkbox" id="${dateDir}-${sessionDir}-im"><label class="form-check-label" for="${dateDir}-${sessionDir}-im">${sessionDir}</label></div>`;
            });
            sessionsHTML += `</div>`;
            treeContainer.innerHTML += dateHTML + sessionsHTML;
        }
        inferenceBsModal?.show();
    } else {
        showErrorOnLabelTrainPage('No recordings found to run inference on.');
    }
}

/** Submits the job to start classification (inference). */
async function submitStartClassification() {
    const datasetNameForModel = document.getElementById('im-dataset').innerText;
    const selectedRecs = [];
    Object.keys(recordingDirTree).forEach(dir => {
        const dirCheckbox = document.getElementById(`${dir}-im`);
        if (dirCheckbox?.checked) {
            selectedRecs.push(dir);
        } else {
            recordingDirTree[dir]?.forEach(subdir => {
                const subdirCheckbox = document.getElementById(`${dir}-${subdir}-im`);
                if (subdirCheckbox?.checked) {
                    selectedRecs.push(`${dir}/${subdir}`);
                }
            });
        }
    });

    if (selectedRecs.length === 0) { showErrorOnLabelTrainPage('No recordings selected for inference.'); return; }
    
    updateTrainingStatusOnUI(datasetNameForModel, "Inference tasks queued...");
    await eel.start_classification(datasetNameForModel, selectedRecs)();
    inferenceBsModal?.hide();
}


// =================================================================
// MAIN APPLICATION LOGIC & EVENT LISTENERS
// =================================================================

/** Fetches and renders the initial list of dataset cards. */
async function loadInitialDatasetCards() {
    try {
        const datasets = await eel.load_dataset_configs()();
        const container = document.getElementById('dataset-container');
        if (!container) return;
        container.className = 'row g-3'; // Add gutter spacing for cards
        let htmlContent = '';

        if (await eel.model_exists("JonesLabModel")()) {
            htmlContent += `<div class="col-md-6 col-lg-4"><div class="card shadow h-100"><div class="card-header bg-dark text-white"><h5 class="card-title mb-0">JonesLabModel <span class="badge bg-info">Default</span></h5></div><div class="card-body d-flex flex-column"><p class="card-text small text-muted">Pre-trained model for general inference.</p></div><div class="card-footer text-end"><button class="btn btn-sm btn-outline-warning" type="button" onclick="showInferenceModal('JonesLabModel')">Infer</button></div></div></div>`;
        }

        if (datasets) {
            for (const datasetName in datasets) {
                if (datasetName === "JonesLabModel") continue;
                const config = datasets[datasetName];
                const behaviors = config.behaviors || [];
                const metrics = config.metrics || {};
                const modelExists = !!config.model;
                const metricHeaders = ['Train #', 'Test #', 'Precision', 'Recall', 'F1 Score'];
                htmlContent += `<div class="col-md-6 col-lg-4"><div class="card shadow h-100"><div class="card-header bg-dark text-white"><h5 class="card-title mb-0">${datasetName}</h5></div><div class="card-body" style="font-size: 0.85rem;">`;
                if (behaviors.length > 0) {
                    htmlContent += `<div class="table-responsive"><table class="table table-sm table-hover small"><thead><tr><th>Behavior</th>${metricHeaders.map(h => `<th class="text-center">${h}</th>`).join('')}</tr></thead><tbody>`;
                    behaviors.forEach(behaviorName => {
                        const bMetrics = metrics[behaviorName] || {};
                        const idSuffixes = ['train-count', 'test-count', 'precision', 'recall', 'fscore'];
                        htmlContent += `<tr><td>${behaviorName}</td>${metricHeaders.map((mh, i) => `<td class="text-center" id="${datasetName}-${behaviorName}-${idSuffixes[i]}">${bMetrics[mh] ?? 'N/A'}</td>`).join('')}</tr>`;
                    });
                    htmlContent += `</tbody></table></div>`;
                } else {
                    htmlContent += `<p class="text-muted">No behaviors defined yet.</p>`;
                }
                htmlContent += `<div class="progress mt-2" id="progress-container-${datasetName}" style="height: 20px; display: none;"><div class="progress-bar progress-bar-striped progress-bar-animated" id="progress-bar-${datasetName}" role="progressbar" style="width: 0%;"></div></div>`;
                htmlContent += `<div id="dataset-status-${datasetName}" class="mt-2 small text-info"></div>`;
                htmlContent += `</div><div class="card-footer text-end"><button class="btn btn-sm btn-outline-primary me-1" type="button" onclick="showPreLabelOptions('${datasetName}')">Label</button><button class="btn btn-sm btn-outline-success me-1" type="button" onclick="showTrainModal('${datasetName}')">Train</button>`;
                if (modelExists) {
                    htmlContent += `<button class="btn btn-sm btn-outline-warning" type="button" onclick="showInferenceModal('${datasetName}')">Infer</button>`;
                }
                htmlContent += `</div></div></div>`;
            }
        }
        container.innerHTML = htmlContent || "<p class='text-light'>No datasets found. Click '+' to create one.</p>";
    } catch (error) {
        console.error("Error loading initial dataset configs:", error);
    }
}

/** Handles timeline scrubbing via mouse drag. */
function handleMouseMoveForLabelScrub(event) {
    const imageElement = event.target;
    if (!imageElement) return;
    const imageRect = imageElement.getBoundingClientRect();
    const x = event.clientX - imageRect.left;
    eel.handle_click_on_label_image(x, 0)?.();
}

/** Cleans up timeline scrubbing event listeners. */
function handleMouseUpForLabelScrub() {
    document.removeEventListener('mousemove', handleMouseMoveForLabelScrub);
    document.removeEventListener('mouseup', handleMouseUpForLabelScrub);
}

/** Updates child checkboxes state when a parent is clicked. */
function updateChildrenCheckboxes(parentCheckboxId, isInfModalSuffix = false) {
    const baseParentDirId = isInfModalSuffix ? parentCheckboxId.slice(0, -3) : parentCheckboxId;
    const subdirs = recordingDirTree[baseParentDirId];
    const parentCheckbox = document.getElementById(parentCheckboxId);
    if (subdirs && parentCheckbox) {
        subdirs.forEach(subdir => {
            const childCheckboxId = isInfModalSuffix ? `${baseParentDirId}-${subdir}-im` : `${baseParentDirId}-${subdir}`;
            const childCheckbox = document.getElementById(childCheckboxId);
            if (childCheckbox) childCheckbox.checked = parentCheckbox.checked;
        });
    }
}

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

// --- Global Event Listeners ---

/** Attaches mousedown listener for timeline scrubbing. */
const labelImageDisplayElement = document.getElementById('label-image');
if (labelImageDisplayElement) {
    labelImageDisplayElement.addEventListener('mousedown', function (event) {
        const imageRect = event.target.getBoundingClientRect();
        const y = event.clientY - imageRect.top;
        const timelineHeightApproximation = 50;
        if (y > (imageRect.height - timelineHeightApproximation - 10) && y < (imageRect.height - 5)) {
            const x = event.clientX - imageRect.left;
            eel.handle_click_on_label_image(x, y)?.();
            document.addEventListener('mousemove', handleMouseMoveForLabelScrub);
            document.addEventListener('mouseup', handleMouseUpForLabelScrub);
        }
    });
}

/** Global keydown listener for all labeling shortcuts. */
window.addEventListener("keydown", (event) => {
    if (labelingInterfaceActive && document.getElementById('label')?.style.display === 'flex') {
        let handled = true;
        if (event.key === "Tab") {
            event.preventDefault();
            jumpToInstance(event.shiftKey ? -1 : 1);
            return;
        }
        switch (event.key) {
            case "ArrowLeft":
                (event.ctrlKey || event.metaKey) ? eel.next_video(-1)() : eel.next_frame(-scrubSpeedMultiplier)();
                break;
            case "ArrowRight":
                (event.ctrlKey || event.metaKey) ? eel.next_video(1)() : eel.next_frame(scrubSpeedMultiplier)();
                break;
            case "ArrowUp":
                scrubSpeedMultiplier = Math.min(scrubSpeedMultiplier * 2, 128);
                break;
            case "ArrowDown":
                scrubSpeedMultiplier = Math.max(1, Math.trunc(scrubSpeedMultiplier / 2));
                break;
            case "Delete":
                eel.delete_instance_from_buffer()();
                break;
            case "Backspace":
                eel.pop_instance_from_buffer()();
                break;
            default:
                let bIdx = -1;
                if (event.keyCode >= 49 && event.keyCode <= 57) bIdx = event.keyCode - 49;
                else if (event.keyCode >= 65 && event.keyCode <= 90) bIdx = event.keyCode - 65 + 9;
                if (bIdx !== -1) { eel.label_frame(bIdx)(); } else { handled = false; }
                break;
        }
        if (handled) event.preventDefault();
    }
});

/** Page unload listeners to ensure Python processes are killed. */
window.addEventListener("unload", function () {
    if (!routingInProgress) { eel.kill_streams()?.().catch(err => console.error(err)); }
});
window.onbeforeunload = function () {
    if (!routingInProgress) { eel.kill_streams()?.().catch(err => console.error(err)); }
};

/** Initial page load setup. */
document.addEventListener('DOMContentLoaded', async () => {
    await waitForEelConnection();
    loadInitialDatasetCards();

    // Assign onclick handlers for modal submission buttons.
    document.getElementById('createDatasetButton')?.addEventListener('click', submitCreateDataset);
    document.getElementById('trainModelButton')?.addEventListener('click', submitTrainModel);
    document.getElementById('startClassificationButton')?.addEventListener('click', submitStartClassification);
});