/**
 * @file startup_page.js
 * Handles the logic for the project startup page, including creating new projects
 * and opening existing ones. Interacts with Electron's IPC for file dialogs
 * and uses Eel to communicate with the Python backend.
 */

// Array of fun blurbs and corresponding links to display on the startup page
const blurbs = [
    "eel included",
    "electronic!",
    "thanks Yann",
    "just lin alg",
    "add <i>Mus musculus</i>",
    "around the clock",
];

const links = [
    "https://github.com/python-eel/Eel",
    "https://www.electronjs.org/",
    "https://ai.meta.com/blog/self-supervised-learning-the-dark-matter-of-intelligence/",
    "https://en.wikipedia.org/wiki/Linear_algebra",
    "https://en.wikipedia.org/wiki/Laboratory_mouse",
    "https://en.wikipedia.org/wiki/Circadian_rhythm",
];

let funBlurbElement = document.getElementById("fun-blurb-text");
if (funBlurbElement) {
    let randomIndex = Math.floor(Math.random() * blurbs.length);
    funBlurbElement.innerHTML = blurbs[randomIndex];
    funBlurbElement.href = links[randomIndex];
}

// IPC renderer for communication with Electron main process (e.g., for file dialogs)
const ipc = window.ipcRenderer; // Ensure this is available if running in Electron

// DOM Elements
const createProjectButton = document.getElementById("create");
const openProjectButton = document.getElementById("open");
const errorModalElement = document.getElementById("errorModal");
const errorMessageElement = document.getElementById("error-message");
let bootstrapErrorModal = errorModalElement ? new bootstrap.Modal(errorModalElement) : null;


/**
 * Flag to distinguish file picker mode:
 * 0 for selecting a parent directory to create a new project.
 * 1 for selecting an existing project directory to open.
 * @type {number}
 */
let filePickerMode = 0;

/**
 * Flag to prevent `kill_streams` from being called during intentional page navigation.
 * @type {boolean}
 */
let routingInProgress = false;

/** Shows the error modal with a custom message. */
function showErrorOnStartup(message) {
    if (errorMessageElement && bootstrapErrorModal) {
        errorMessageElement.innerText = message;
        bootstrapErrorModal.show();
    } else {
        alert(message); // Fallback if Bootstrap modal isn't available
    }
}

/** Navigates to the record page. */
function routeToRecordPage() {
    routingInProgress = true;
    window.location.href = "./record.html";
}

// Event listener for the "Create Project" button
if (createProjectButton) {
    createProjectButton.addEventListener("click", function () {
        filePickerMode = 0;
        if (ipc) { // Send IPC message only if ipc is available (Electron context)
            ipc.send("open-file-dialog");
        } else {
            showErrorOnStartup("File dialog functionality is not available (not in Electron?).");
        }
    });
}

// Event listener for the "Open Project" button
if (openProjectButton) {
    openProjectButton.addEventListener("click", function () {
        filePickerMode = 1;
        if (ipc) {
            ipc.send("open-file-dialog");
        } else {
            showErrorOnStartup("File dialog functionality is not available (not in Electron?).");
        }
    });
}

/**
 * Handles the creation of a new project after the user has selected a parent directory
 * and entered a project name in the modal.
 */
async function handleCreateProjectSubmit() {
    const parentDirDisplay = document.getElementById("parent-directory");
    const projectNameInput = document.getElementById("project-name");

    if (!parentDirDisplay || !projectNameInput) {
        showErrorOnStartup("Internal UI error: Create project modal elements are missing.");
        return;
    }

    let parentDirectoryPath = parentDirDisplay.innerText; // Use innerText as it's displaying the path
    let projectNameValue = projectNameInput.value;

    if (!projectNameValue.trim()) {
        showErrorOnStartup("Project name cannot be empty.");
        return;
    }

    try {
        console.log(`Attempting to create project: ${projectNameValue} in ${parentDirectoryPath}`);
        const result = await eel.create_project(parentDirectoryPath, projectNameValue)();
        const projectCreated = result[0];
        const projectDetailsDict = result[1];

        if (projectCreated && projectDetailsDict) {
            window.localStorage.setItem("project", JSON.stringify(projectDetailsDict));
            console.log("Project created and stored in localStorage. Routing to record page.");
            routeToRecordPage();
        } else {
            showErrorOnStartup("Project could not be created. It might already exist or the name is invalid.");
        }
    } catch (error) {
        console.error("Error during eel.create_project call:", error);
        showErrorOnStartup(`An error occurred: ${error.message || "Unknown error creating project."}`);
    }
}

// Listener for the 'selected-directory' event from the Electron main process (file dialog response)
if (ipc) {
    ipc.on("selected-directory", async (_event, selectedPath) => {
        if (selectedPath == null) { // User cancelled the dialog
            console.log("Directory selection cancelled.");
            return;
        }
        console.log(`Directory selected: ${selectedPath}, mode: ${filePickerMode}`);

        if (filePickerMode === 0) { // Create project mode: path is parent directory
            const createModalElement = document.getElementById("createModal");
            const parentDirDisplayElement = document.getElementById("parent-directory");
            if (createModalElement && parentDirDisplayElement) {
                let modal = new bootstrap.Modal(createModalElement);
                parentDirDisplayElement.innerText = selectedPath;
                document.getElementById("project-name").value = ""; // Clear previous project name
                modal.show();
                // Actual project creation is triggered by a button within this modal that calls handleCreateProjectSubmit()
            } else {
                console.error("Create project modal elements not found.");
            }
        } else { // Open project mode (filePickerMode === 1): path is project directory
            try {
                console.log(`Attempting to load project: ${selectedPath}`);
                const result = await eel.load_project(selectedPath)();
                const projectLoaded = result[0];
                const projectDetailsDict = result[1];

                if (projectLoaded && projectDetailsDict) {
                    window.localStorage.setItem("project", JSON.stringify(projectDetailsDict));
                    console.log("Project loaded and stored in localStorage. Routing to record page.");
                    routeToRecordPage();
                } else {
                    showErrorOnStartup("Selected directory is not a valid CBAS project or could not be loaded.");
                }
            } catch (error) {
                console.error("Error during eel.load_project call:", error);
                showErrorOnStartup(`An error occurred while loading the project: ${error.message || "Unknown error loading project."}`);
            }
        }
    });
}


/**
 * Event listener for page unload. Attempts to call a Python function to clean up
 * (e.g., stop FFMPEG streams) if the unload is not due to intentional routing.
 */
window.addEventListener("unload", function () {
    if (!routingInProgress) {
        console.log("StartupPage: Unload event (not routing). Attempting to call eel.kill_streams.");
        if (typeof eel !== 'undefined' && typeof eel.kill_streams === 'function') {
            eel.kill_streams()().catch(err => console.error("Error in eel.kill_streams on unload:", err));
        }
    }
});

/**
 * Fallback for `onbeforeunload` to ensure cleanup attempt.
 */
window.onbeforeunload = function () {
    if (!routingInProgress) {
        console.log("StartupPage: Beforeunload event (not routing). Attempting to call eel.kill_streams.");
        if (typeof eel !== 'undefined' && typeof eel.kill_streams === 'function') {
            eel.kill_streams()().catch(err => console.error("Error in eel.kill_streams on beforeunload:", err));
        }
    }
    // To prevent the default browser confirmation dialog, return undefined or nothing.
    // If you want a confirmation, return a string: return "Are you sure you want to leave?";
};

// Assign createProject to the modal's save button if it exists
const createProjectModalButton = document.querySelector('#createModal .btn-primary');
if (createProjectModalButton) {
    createProjectModalButton.onclick = handleCreateProjectSubmit;
}