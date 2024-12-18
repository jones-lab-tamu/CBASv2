blurbs = [
    "eel included",
    "electronic!",
    "thanks Yann",
    "just lin alg",
    "add <i>Mus musculus</i>",
    "around the clock",
];

links = [
    "https://github.com/python-eel/Eel",
    "https://www.electronjs.org/",
    "https://ai.meta.com/blog/self-supervised-learning-the-dark-matter-of-intelligence/",
    "https://en.wikipedia.org/wiki/Linear_algebra",
    "https://en.wikipedia.org/wiki/Laboratory_mouse",
    "https://en.wikipedia.org/wiki/Circadian_rhythm",
];

let fb = document.getElementById("fun-blurb-text");

let index = Math.floor(Math.random() * blurbs.length);
fb.innerHTML = blurbs[index];
fb.href = links[index];

const ipc = window.ipcRenderer;

const createBtn = document.getElementById("create");
const openBtn = document.getElementById("open");

/* We use the same ipc file picker for both creating and opening a 
 * project, so we distinguish which one we are doing with this flag
 */
let filePickerMode = 0; // 0 for create, 1 for open

let routing = false;

function routeRecord() {
    routing = true;
    window.open("./record.html", "_self");
}
function routeLabelTrain() {
    routing = true;
    window.open("./label-train.html", "_self");
}

function routeVisualize() {
    routing = true;
    window.open("./visualize.html", "_self");
}

createBtn.addEventListener("click", function (event) {
    filePickerMode = 0;
    ipc.send("open-file-dialog");
});

openBtn.addEventListener("click", function (event) {
    filePickerMode = 1;
    ipc.send("open-file-dialog");
});

async function createProject() {
    let parentDirectory = document.getElementById("parent-directory").innerHTML;
    let projectName = document.getElementById("project-name").value;

    let result = await eel.create_project(parentDirectory, projectName)();

    let projectExists = result[0];
    let dict = result[1];

    if (projectExists) {
        let project_dict = dict;
        window.localStorage.setItem("project", JSON.stringify(project_dict));
        routeRecord();
    } else {
        let modal = new bootstrap.Modal(document.getElementById("errorModal"));
        document.getElementById("error-message").innerHTML =
            "Project could not be created. Check if the entered project already exists.";
        modal.show();
    }
}

// This is called when a user selects a directory.
ipc.on("selected-directory", async (_event, path) => {
    // The user cancelled
    if (path == null) {
        return;
    }

    if (filePickerMode == 0) {
        // They selected a parent directory, now let them create a new project
        let modal = new bootstrap.Modal(document.getElementById("createModal"));
        document.getElementById("parent-directory").innerHTML = path;
        modal.show(); // The createProject callback is called from the modal HTML
    } else {
        let result = await eel.load_project(path)();
        let projectExists = result[0];
        let dict = result[1];

        if (projectExists) {
            let project_dict = dict;
            window.localStorage.setItem("project", JSON.stringify(project_dict));
            routeRecord();
        } else {
            // show error modal
            let modal = new bootstrap.Modal(
                document.getElementById("errorModal")
            );
            document.getElementById("error-message").innerHTML =
                "Project does not exist.";
            modal.show();
        }
    }
});

// This is essentially black magic
window.addEventListener("unload", function (e) {
    if (!routing) {
        eel.kill_streams();
    }
});

window.onbeforeunload = function () {
    if (!routing) {
        eel.kill_streams();
    }
};