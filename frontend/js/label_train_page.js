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

/** Debounce timer for confidence filter */ 
let confidenceFilterDebounceTimer;

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
const importVideosModalElement = document.getElementById('importVideosModal');
const manageDatasetModalElement = document.getElementById('manageDatasetModal');

let addDatasetBsModal = addDatasetModalElement ? new bootstrap.Modal(addDatasetModalElement) : null;
let trainBsModal = trainModalElement ? new bootstrap.Modal(trainModalElement) : null;
let inferenceBsModal = inferenceModalElement ? new bootstrap.Modal(inferenceModalElement) : null;
let generalErrorBsModal = errorModalElement ? new bootstrap.Modal(errorModalElement) : null;
let preLabelBsModal = preLabelModalElement ? new bootstrap.Modal(preLabelModalElement) : null;
let importVideosBsModal = importVideosModalElement ? new bootstrap.Modal(importVideosModalElement) : null;
let selectedVideoPathsForImport = [];
let manageDatasetBsModal = manageDatasetModalElement ? new bootstrap.Modal(manageDatasetModalElement) : null;

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

function showManageDatasetModal(datasetName) {
    if (!manageDatasetBsModal) return;

    // Set the dataset name in the modal title
    document.getElementById('md-dataset-name').innerText = datasetName;

    // Find the "Reveal" button and attach the correct Eel call to it
    const revealBtn = document.getElementById('revealFilesButton');
    if (revealBtn) {
        // We set the onclick here to ensure it always has the correct datasetName
        revealBtn.onclick = () => {
            eel.reveal_dataset_files(datasetName)();
            manageDatasetBsModal.hide(); // Hide modal after clicking
        };
    }
    
    manageDatasetBsModal.show();
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
function updateLabelImageSrc(mainFrameBlob, timelineBlob, zoomBlob) {
    const mainFrameImg = document.getElementById('label-image');
    const fullTimelineImg = document.getElementById('full-timeline-image');
    const zoomImg = document.getElementById('zoom-bar-image');
    const blankGif = "data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=";

    if (mainFrameImg) {
        mainFrameImg.src = mainFrameBlob ? "data:image/jpeg;base64," + mainFrameBlob : "assets/noVideo.png";
    }
    
    if (fullTimelineImg) {
        fullTimelineImg.src = timelineBlob ? "data:image/jpeg;base64," + timelineBlob : blankGif;
    }
    
    // The zoom bar is now always visible, so we just update its source.
    if (zoomImg) {
        zoomImg.src = zoomBlob ? "data:image/jpeg;base64," + zoomBlob : blankGif;
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
    controlsHTML += `<li class="list-group-item bg-dark text-light d-flex justify-content-between"><strong>Behavior</strong><span><strong>Confidence</strong></span><span><strong>Key</strong></span><span><strong>Count</strong></span></li>`;

    if (behaviors && colors && behaviors.length === colors.length) {
        behaviors.forEach((behaviorName, index) => {
            const key = (index < 9) ? (index + 1) : String.fromCharCode('a'.charCodeAt(0) + (index - 9));
            const bgColor = colors[index];
            const textColor = getTextColorForBg(bgColor);
            controlsHTML += `
                <li class="list-group-item bg-dark text-light d-flex justify-content-between align-items-center" 
                    id="behavior-row-${behaviorName.replace(/[\W_]+/g, '-')}"
                    onclick="eel.label_frame(${index})()" style="cursor: pointer;" title="Click or press '${key}' to label '${behaviorName}'">
                    <span style="flex-basis: 40%;">${behaviorName}</span>
                    <span class="confidence-badge-placeholder" style="flex-basis: 20%;"></span> 
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
	
    const confidenceSlider = document.getElementById('confidence-slider');
    const sliderValueDisplay = document.getElementById('slider-value-display');
    if (confidenceSlider && sliderValueDisplay) {
        confidenceSlider.value = 100;
        sliderValueDisplay.textContent = '100%';
    }	
	
}

// In frontend/js/label_train_page.js

/** Sets the UI mode for labeling (scratch vs. review) and updates controls text. */
eel.expose(setLabelingModeUI);
function setLabelingModeUI(mode, modelName = '') {
    const controlsHeader = document.querySelector('#controls .card-header');
    const cheatSheet = document.getElementById('labeling-cheat-sheet');
    if (!controlsHeader || !cheatSheet) return;

    if (mode === 'review') {
        // --- CONFIGURE FOR REVIEW MODE ---
        controlsHeader.classList.remove('bg-dark');
        controlsHeader.classList.add('bg-success');
        controlsHeader.querySelector('h5').innerHTML = `Reviewing: <span class="badge bg-light text-dark">${modelName}</span>`;
        
        // This is the updated cheat sheet for Review Mode
        cheatSheet.innerHTML = `
            <div class="card bg-dark">
              <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="mb-0 text-success"><i class="bi bi-robot me-2"></i>Review Mode Controls</h5>
                <div id="confidence-filter-container" class="d-flex align-items-center w-50">
                  <label for="confidence-slider" class="form-label me-3 mb-0 text-nowrap text-light">Filter Confidence < </label>
                  <input type="range" class="form-range" min="0" max="100" step="1" value="100" id="confidence-slider">
                  <span id="slider-value-display" class="ms-3 badge bg-info" style="width: 55px;">100%</span>
                  <button id="reset-slider-btn" class="btn btn-sm btn-outline-secondary ms-2">Reset</button>
                </div>
              </div>
              <div class="card-body" style="font-size: 0.9rem;">
                <div class="row">
                  <div class="col-md-6">
                    <ul class="list-unstyled">
                      <li><kbd>Tab</kbd> / <kbd>Shift+Tab</kbd> : Next/Prev Instance</li>
                      <li><kbd>←</kbd> / <kbd>→</kbd> : Step one frame (within instance)</li>
                      <li><kbd>[</kbd> / <kbd>]</kbd> : Set Start/End of Instance</li>
                      <li><kbd>Enter</kbd> : Confirm / Lock / Unlock Instance</li>
                    </ul>
                  </div>
                  <div class="col-md-6">
                    <ul class="list-unstyled">
                      <li><kbd>1</kbd> - <kbd>9</kbd> : Change Instance Label</li>
                      <li><kbd>Delete</kbd> : Delete instance at current frame</li>
                      <li><kbd>Backspace</kbd> : Undo last added label</li>
                      <li><kbd>Ctrl</kbd> + <kbd>S</kbd> : Commit Corrections</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>`;

    } else { // 'scratch' mode
        // --- CONFIGURE FOR SCRATCH MODE ---
        controlsHeader.classList.remove('bg-success');
        controlsHeader.classList.add('bg-dark');
        controlsHeader.querySelector('h5').innerHTML = `Behaviors (Click to label)`;
        
        // This is the updated cheat sheet for Scratch Mode
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
                      <li><kbd>Ctrl</kbd> + <kbd>S</kbd> : Commit Corrections</li>
                    </ul>
                  </div>
                  <div class="col-md-6">
                    <ul class="list-unstyled">
                      <li><kbd>1</kbd> - <kbd>9</kbd> : Start / End a new label</li>
                      <li><kbd>Delete</kbd> : Delete instance at current frame</li>
                      <li><kbd>Backspace</kbd> : Undo last added label</li>
                      <li><kbd>Ctrl</kbd> + <kbd>←</kbd> / <kbd>→</kbd> : Prev/Next video</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>`;
    }

    // After rebuilding the HTML, we must re-attach the event listener for the slider
    const confidenceSlider = document.getElementById('confidence-slider');
    const sliderValueDisplay = document.getElementById('slider-value-display');
    const resetSliderBtn = document.getElementById('reset-slider-btn');
    const timelineContainer = document.getElementById('full-timeline-section');

    if (confidenceSlider && sliderValueDisplay) {
        confidenceSlider.addEventListener('input', function() {
            sliderValueDisplay.textContent = `${this.value}%`;
            if (timelineContainer) {
                if (parseInt(this.value) < 100) {
                    timelineContainer.classList.add('timeline-filtered');
                } else {
                    timelineContainer.classList.remove('timeline-filtered');
                }
            }
            clearTimeout(confidenceFilterDebounceTimer);
            confidenceFilterDebounceTimer = setTimeout(() => {
                eel.refilter_instances(parseInt(this.value))();
            }, 400);
        });
    }
    if (resetSliderBtn) {
        resetSliderBtn.addEventListener('click', function() {
            if (confidenceSlider && sliderValueDisplay) {
                confidenceSlider.value = 100;
                sliderValueDisplay.textContent = '100%';
                if (timelineContainer) {
                    timelineContainer.classList.remove('timeline-filtered');
                }
                eel.refilter_instances(100)();
            }
        });
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

eel.expose(updateConfidenceBadge);
function updateConfidenceBadge(behaviorName, confidence) {
    // First, clear all existing badges
    document.querySelectorAll('.confidence-badge-placeholder').forEach(el => el.innerHTML = '');

    if (behaviorName && confidence !== null) {
        const rowId = `behavior-row-${behaviorName.replace(/[\W_]+/g, '-')}`;
        const row = document.getElementById(rowId);
        if (!row) return;

        const placeholder = row.querySelector('.confidence-badge-placeholder');
        if (!placeholder) return;

        let badgeClass = 'bg-secondary';
        if (confidence >= 0.9) {
            badgeClass = 'bg-success'; // High confidence
        } else if (confidence >= 0.7) {
            badgeClass = 'bg-warning text-dark'; // Medium confidence
        } else {
            badgeClass = 'bg-danger'; // Low confidence
        }

        const confidencePercent = (confidence * 100).toFixed(0);
        placeholder.innerHTML = `<span class="badge ${badgeClass}">${confidencePercent}%</span>`;
    }
}

// Eel-exposed function to just handle VISUALS
eel.expose(setConfirmationModeUI);
function setConfirmationModeUI(isConfirming) {
    const commitBtn = document.getElementById('save-labels-btn');
    const cancelBtn = document.getElementById('cancel-commit-btn');
    
    if (!commitBtn || !cancelBtn) return;

    if (isConfirming) {
        commitBtn.innerHTML = '<i class="bi bi-check-circle-fill me-2"></i>Confirm & Save';
        commitBtn.classList.remove('btn-success');
        commitBtn.classList.add('btn-primary');
        cancelBtn.style.display = 'inline-block';
    } else {
        commitBtn.innerHTML = '<i class="bi bi-save-fill me-2"></i>Commit Corrections';
        commitBtn.classList.remove('btn-primary');
        commitBtn.classList.add('btn-success');
        cancelBtn.style.display = 'none';
    }
}

// =================================================================
// UI INTERACTION & EVENT HANDLERS (Called FROM HTML)
// =================================================================

function jumpToFrame() {
    const input = document.getElementById('frame-jump-input');
    if (input && input.value) {
        const frameNum = parseInt(input.value);
        if (!isNaN(frameNum)) {
            eel.jump_to_frame(frameNum)();
        }
    }
}
// State machine function called by the main button
function handleCommitClick() {
    const commitBtn = document.getElementById('save-labels-btn');
    // Check the button's current text to determine the state
    if (commitBtn.innerText.includes("Commit Corrections")) {
        // If it's in the initial state, call the staging function
        eel.stage_for_commit()();
    } else {
        // Otherwise, it's in confirmation mode, so show the dialog and save
        if (confirm("Are you sure you want to commit these verified labels? This will overwrite previous labels for this video file.")) {
            eel.save_session_labels()();
        }
    }
}
/** Jumps to the next or previous labeled instance in the video. */
function jumpToInstance(direction) {
    eel.jump_to_instance(direction)();
}

/** Opens the 'Labeling Options' modal and populates its fields. */
async function showPreLabelOptions(datasetName) {
    // --- 1. Get references to all modal elements ---
    const modelSelect = document.getElementById('pl-model-select');
    const sessionSelect = document.getElementById('pl-session-select');
    const videoSelect = document.getElementById('pl-video-select');
    const infoDiv = document.getElementById('pl-behavior-match-info');

    // --- 2. Reset the modal to its initial state ---
    document.getElementById('pl-dataset-name').innerText = datasetName;
    modelSelect.innerHTML = '<option selected disabled>Loading...</option>';
    sessionSelect.innerHTML = '<option selected disabled>First, choose a model...</option>';
    videoSelect.innerHTML = '<option selected disabled>First, choose a session...</option>';
    infoDiv.innerHTML = '';
    modelSelect.disabled = true;
    sessionSelect.disabled = true;
    videoSelect.disabled = true;

    // --- 3. Fetch all necessary data from Python backend ---
    const allDatasetConfigs = await eel.load_dataset_configs()();
    const allModelConfigs = await eel.get_model_configs()();
    const allModels = await eel.get_available_models()();
    
    const targetDatasetConfig = allDatasetConfigs[datasetName];
    if (!targetDatasetConfig || !targetDatasetConfig.behaviors) {
        showErrorOnLabelTrainPage(`Could not load behaviors for dataset '${datasetName}'.`);
        return;
    }
    const targetBehaviors = new Set(targetDatasetConfig.behaviors);

    // --- 4. Populate the first dropdown with compatible models ---
    let compatibleModels = [];
    if (allModels && allModelConfigs) {
        allModels.forEach(modelName => {
            const modelConfig = allModelConfigs[modelName];
            if (modelConfig && modelConfig.behaviors) {
                const modelBehaviors = new Set(modelConfig.behaviors);
                if ([...targetBehaviors].some(b => modelBehaviors.has(b))) {
                    compatibleModels.push(modelName);
                }
            }
        });
    }

    // --- 5. Update the UI based on whether compatible models were found ---
    if (compatibleModels.length > 0) {
        modelSelect.innerHTML = '<option selected disabled>Choose a model...</option>';
        compatibleModels.forEach(name => modelSelect.innerHTML += `<option value="${name}">${name}</option>`);
        modelSelect.disabled = false;
    } else {
        modelSelect.innerHTML = '<option selected disabled>No compatible models found</option>';
    }

    // --- 6. Show the modal ---
    preLabelBsModal.show();
}

// --- Event listeners need to be attached ONCE when the page loads ---
async function onModelSelectChange(event) {
    const modelName = event.target.value;
    const datasetName = document.getElementById('pl-dataset-name').innerText;
    
    // --- UI Nudging ---
    const infoDiv = document.getElementById('pl-behavior-match-info');
    infoDiv.innerHTML = ''; // Clear previous
    const allModelConfigs = await eel.get_model_configs()(); // Re-fetch or pass down
    const allDatasetConfigs = await eel.load_dataset_configs()();
    const targetBehaviors = new Set(allDatasetConfigs[datasetName].behaviors);
    const modelConfig = allModelConfigs[modelName];

    if (modelConfig && modelConfig.behaviors) {
        const modelBehaviors = new Set(modelConfig.behaviors);
        const matching = [...targetBehaviors].filter(b => modelBehaviors.has(b)).join(', ');
        const nonMatching = [...targetBehaviors].filter(b => !modelBehaviors.has(b)).join(', ');
        infoDiv.innerHTML = `Will pre-label for: <strong>${matching}</strong>.`;
        if (nonMatching) {
            infoDiv.innerHTML += `<br><span class="text-warning">Will ignore: ${nonMatching}</span>`;
        }
    }

    // --- Populate Session Dropdown ---
    const sessionSelect = document.getElementById('pl-session-select');
    sessionSelect.disabled = false;
    sessionSelect.innerHTML = '<option>Loading sessions...</option>';
    const sessions = await eel.get_inferred_session_dirs(datasetName, modelName)();
    
    sessionSelect.innerHTML = '<option selected disabled>Choose a session...</option>';
    if (sessions && sessions.length > 0) {
        sessions.forEach(dir => sessionSelect.innerHTML += `<option value="${dir}">${dir}</option>`);
    } else {
        sessionSelect.innerHTML = '<option selected disabled>No inferred sessions</option>';
        sessionSelect.disabled = true;
    }
}

async function onSessionSelectChange(event) {
    const sessionDir = event.target.value;
    const modelName = document.getElementById('pl-model-select').value;
    const videoSelect = document.getElementById('pl-video-select');

    videoSelect.disabled = false;
    videoSelect.innerHTML = '<option>Loading videos...</option>';
    const videos = await eel.get_inferred_videos_for_session(sessionDir, modelName)();

    videoSelect.innerHTML = '<option selected disabled>Choose a video...</option>';
    if (videos && videos.length > 0) {
        videos.forEach(v => videoSelect.innerHTML += `<option value="${v[0]}">${v[1]}</option>`);
    } else {
        videoSelect.innerHTML = '<option selected disabled>No videos found</option>';
        videoSelect.disabled = true;
    }
}

/** Triggers a backend refresh of all project data and re-renders the dataset cards. */
async function refreshAllDatasets() {
    console.log("Refreshing datasets from disk...");
    const spinner = document.getElementById('cover-spin');
    if (spinner) spinner.style.visibility = 'visible';

    try {
        // Call a new, simple backend function to trigger a project-wide reload
        await eel.reload_project_data()();
        
        // Now, simply call the existing function to redraw the UI with the new data
        await loadInitialDatasetCards();
    } catch (error) {
        console.error("Failed to refresh datasets:", error);
        showErrorOnLabelTrainPage("An error occurred while trying to refresh the datasets.");
    } finally {
        if (spinner) spinner.style.visibility = 'hidden';
    }
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

/** Starts the import process */
async function showImportVideosDialog() {
    if (!window.electronAPI) {
        showErrorOnLabelTrainPage("The file system API is not available.");
        return;
    }
    try {
        // Use 'invoke' to get the file paths back directly from the main process
        const filePaths = await window.electronAPI.invoke('show-open-video-dialog');

        if (filePaths && filePaths.length > 0) {
            selectedVideoPathsForImport = filePaths;
            document.getElementById('import-file-count').textContent = filePaths.length;
            document.getElementById('import-session-name').value = ''; // Clear previous
            importVideosBsModal?.show();
        } else {
            console.log("User cancelled video selection.");
        }
    } catch (err) {
        showErrorOnLabelTrainPage("Could not open file dialog: " + err.message);
    }
}

/** Handles the modal submission */
async function handleImportSubmit() {
    const sessionName = document.getElementById('import-session-name').value;
    if (!sessionName.trim()) {
        showErrorOnLabelTrainPage("Session Name cannot be empty.");
        return;
    }
    if (selectedVideoPathsForImport.length === 0) {
        showErrorOnLabelTrainPage("No video files were selected.");
        return;
    }

    importVideosBsModal?.hide();
    // General loading spinner
    document.getElementById('cover-spin').style.visibility = 'visible';

    try {
        // Call the new backend function
        const success = await eel.import_videos(sessionName, selectedVideoPathsForImport)();
        if (success) {
            // IMPORTANT: Reload the dataset cards to reflect the new recordings
            await loadInitialDatasetCards();
            // Potentially show a success message
        } else {
            showErrorOnLabelTrainPage("Failed to import videos. The session name might already exist or an error occurred on the backend.");
        }
    } catch (error) {
        showErrorOnLabelTrainPage("An error occurred during import: " + error.message);
    } finally {
        // Hide spinner
        document.getElementById('cover-spin').style.visibility = 'hidden';
    }
}

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

        // --- Renders the card for the default JonesLabModel ---
        if (await eel.model_exists("JonesLabModel")()) {
            htmlContent += `
                <div class="col-md-6 col-lg-4">
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

        // --- Renders cards for all user-created datasets ---
        if (datasets) {
            for (const datasetName in datasets) {
                // Skip rendering a dataset card if it's the default model, as it's handled above
                if (datasetName === "JonesLabModel") continue;
                
                const config = datasets[datasetName];
                const behaviors = config.behaviors || [];
                const metrics = config.metrics || {};
                const modelExists = !!config.model;
                const metricHeaders = ['Train #', 'Test #', 'Precision', 'Recall', 'F1 Score'];

                htmlContent += `
                    <div class="col-md-6 col-lg-4">
                        <div class="card shadow h-100">
                            <div class="card-header bg-dark text-white">
                                <h5 class="card-title mb-0">${datasetName}</h5>
                            </div>
                            <div class="card-body" style="font-size: 0.85rem;">`;

                if (behaviors.length > 0) {
                    htmlContent += `
                        <div class="table-responsive">
                            <table class="table table-sm table-hover small">
                                <thead>
                                    <tr>
                                        <th>Behavior</th>
                                        ${metricHeaders.map(h => `<th class="text-center">${h}</th>`).join('')}
                                    </tr>
                                </thead>
                                <tbody>`;

                    behaviors.forEach(behaviorName => {
                        const bMetrics = metrics[behaviorName] || {};
                        const idSuffixes = ['train-count', 'test-count', 'precision', 'recall', 'fscore'];
                        htmlContent += `
                            <tr>
                                <td>${behaviorName}</td>
                                ${metricHeaders.map((mh, i) => `<td class="text-center" id="${datasetName}-${behaviorName}-${idSuffixes[i]}">${bMetrics[mh] ?? 'N/A'}</td>`).join('')}
                            </tr>`;
                    });

                    htmlContent += `
                                </tbody>
                            </table>
                        </div>`;
                } else {
                    htmlContent += `<p class="text-muted">No behaviors defined yet.</p>`;
                }

                // Progress bar and status text
                htmlContent += `
                    <div class="progress mt-2" id="progress-container-${datasetName}" style="height: 20px; display: none;">
                        <div class="progress-bar progress-bar-striped progress-bar-animated" id="progress-bar-${datasetName}" role="progressbar" style="width: 0%;"></div>
                    </div>
                    <div id="dataset-status-${datasetName}" class="mt-2 small text-info"></div>
                </div>`;

                // Card Footer with all buttons
                htmlContent += `
                    <div class="card-footer d-flex justify-content-end align-items-center">
                        <!-- NEW "Manage" button, aligned to the left -->
                        <button class="btn btn-sm btn-outline-secondary me-auto" type="button" onclick="showManageDatasetModal('${datasetName}')" title="Manage dataset files">
                            <i class="bi bi-folder2-open"></i> Manage
                        </button>
                        
                        <!-- Existing action buttons, aligned to the right -->
                        <button class="btn btn-sm btn-outline-primary me-1" type="button" onclick="showPreLabelOptions('${datasetName}')">Label</button>
                        <button class="btn btn-sm btn-outline-success me-1" type="button" onclick="showTrainModal('${datasetName}')">Train</button>`;
                
                if (modelExists) {
                    htmlContent += `<button class="btn btn-sm btn-outline-warning" type="button" onclick="showInferenceModal('${datasetName}')">Infer</button>`;
                }

                htmlContent += `
                            </div>
                        </div>
                    </div>`;
            }
        }

        // Final check and render
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
const fullTimelineElement = document.getElementById('full-timeline-image');
if (fullTimelineElement) {
    fullTimelineElement.addEventListener('mousedown', function (event) {
        const x = event.offsetX; // offsetX is simpler when bound to the element
        eel.handle_click_on_label_image(x, 0)?.();
        document.addEventListener('mousemove', handleMouseMoveForLabelScrub);
        document.addEventListener('mouseup', handleMouseUpForLabelScrub);
    });
}

const zoomBarImageElement = document.getElementById('zoom-bar-image');
if (zoomBarImageElement) {
    zoomBarImageElement.addEventListener('mousedown', function (event) {
        const imageRect = event.target.getBoundingClientRect();
        const x = event.clientX - imageRect.left;
        eel.get_zoom_range_for_click(x)();
    });
}

/** Global keydown listener for all labeling shortcuts. */
window.addEventListener("keydown", (event) => {
    // Ignore key presses if a modal is open or if the labeling UI isn't active
    if (document.querySelector('.modal.show') || !labelingInterfaceActive || document.getElementById('label')?.style.display !== 'flex') {
        return;
    }

    // --- SAVE HOTKEY ---
    if (event.ctrlKey && event.key.toLowerCase() === 's') {
        event.preventDefault(); // Prevent the browser's default "Save Page" dialog
        const commitBtn = document.getElementById('save-labels-btn');
        if (commitBtn) {
            commitBtn.click(); // Programmatically click the main commit/save button
        }
        return; // Stop further processing for this event
    }

    // Ignore other key presses if the user is typing in the jump-to-frame input
    if (document.activeElement === document.getElementById('frame-jump-input')) {
        // Allow 'Enter' to trigger the jump
        if (event.key === 'Enter') {
            jumpToFrame();
        }
        return;
    }
    
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
        case "[":
            eel.update_instance_boundary('start')();
            break;
        case "]":
            eel.update_instance_boundary('end')();
            break;
        case "Enter":
            eel.confirm_selected_instance()();
            break;
        default:
            let bIdx = -1;
            if (event.keyCode >= 49 && event.keyCode <= 57) bIdx = event.keyCode - 49;
            else if (event.keyCode >= 65 && event.keyCode <= 90) bIdx = event.keyCode - 65 + 9;
            if (bIdx !== -1) { eel.label_frame(bIdx)(); } else { handled = false; }
            break;
    }

    if (handled) {
        event.preventDefault();
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
    document.getElementById('modal-import-button-final')?.addEventListener('click', handleImportSubmit);
	document.getElementById('pl-model-select')?.addEventListener('change', onModelSelectChange);
    document.getElementById('pl-session-select')?.addEventListener('change', onSessionSelectChange);

    const confidenceSlider = document.getElementById('confidence-slider');
    const sliderValueDisplay = document.getElementById('slider-value-display');
    if (confidenceSlider && sliderValueDisplay) {
        confidenceSlider.addEventListener('input', function() {
            // Update the percentage display immediately
            sliderValueDisplay.textContent = `${this.value}%`;

			const timelineContainer = document.getElementById('full-timeline-section');
			if (timelineContainer) {
				if (parseInt(this.value) < 100) {
					timelineContainer.classList.add('timeline-filtered');
				} else {
					timelineContainer.classList.remove('timeline-filtered');
				}
			}

            // Debounce the call to the backend
            clearTimeout(confidenceFilterDebounceTimer);
            confidenceFilterDebounceTimer = setTimeout(() => {
                eel.refilter_instances(parseInt(this.value))();
            }, 400); // 400ms delay
        });
    }

	const resetSliderBtn = document.getElementById('reset-slider-btn');
	if (resetSliderBtn) {
		resetSliderBtn.addEventListener('click', function() {
			const slider = document.getElementById('confidence-slider');
			const display = document.getElementById('slider-value-display');
			const timelineContainer = document.getElementById('full-timeline-section');
			
			if (slider && display) {
				slider.value = 100;
				display.textContent = '100%';

				// Explicitly remove the 'timeline-filtered' class when resetting.
				if (timelineContainer) {
					timelineContainer.classList.remove('timeline-filtered');
				}
				
				// Trigger the filter update
				eel.refilter_instances(100)();
			}
		});
	}

});