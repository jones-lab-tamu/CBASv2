const ipc = window.ipcRenderer

let loaded = false
let routing = false

let labeling = false

let datasetModal = new bootstrap.Modal(document.getElementById('addDataset'))

let trainModal = new bootstrap.Modal(document.getElementById('trainModal'))

let inferenceModal = new bootstrap.Modal(document.getElementById('inferenceModal'))

let dir_children = new Object();

let surf = 1;

function read(attr) {
    let project_string = window.localStorage.getItem('project')
    let project = JSON.parse(project_string)
    return project[attr]
}

function routeRecord() {
    routing = true
    window.open('./record.html', '_self');
}

function routeLabelTrain() {
    routing = true
    window.open('./label-train.html', '_self');
}

function routeVisualize() {
    routing = true
    window.open('./visualize.html', '_self');
}

function handleMouseMove(event) {
    let x = event.offsetX;
    let y = event.offsetY;
    eel.handle_click_on_label_image(x, y)
}

function handleMouseUp() {
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
}

/* A user clicking initiates on 'label-image' causes CBAS to begin tracking cursor movement and clicks. */
document.getElementById('label-image').addEventListener('mousedown', function (event) {
    let x = event.offsetX;
    let y = event.offsetY;

    if (y > 500) {
        eel.handle_click_on_label_image(x, y)

        document.addEventListener('mousemove', handle_mousemove);
        document.addEventListener('mouseup', handle_mouseup);
    }
});

async function loadInitial() {
    let datasets = await eel.load_dataset_configs()();

    let container = document.getElementById('dataset-container')

    container.innerHTML = `
            <div class="row-auto"  style="padding:15px">
                <div class="card shadow text-light bg-light" style="width: 700px; height:max-content;border-radius:1px;">
                    <div class="card-header bg-dark" style="border-radius:1px;">
                        <h1 class="display-6"> JonesLabModel </h1>
                        <button class="btn btn-outline-warning position-absolute top-0 end-0" style='width:80px;margin-right:15px;margin-top:15px' type="submit" onclick="showInferenceModal('JonesLabModel')">Infer</button>
                    </div>
                </div>
            </div>`

    let names = Object.keys(datasets)
    for (i = 0; i < names.length; i++) {
        let n = names[i]
        let name = datasets[n]['name']

        let behaviors = datasets[n]['behaviors']

        let metrics = datasets[n]['metrics']

        let mheaders = ['Train #', 'Test #', 'Precision', 'Recall', 'F1 Score']

        let html = `

                            <div class="row-auto"  style="padding:15px">
                                <div class="card shadow text-light bg-light" style="width: 700px; height:max-content;">
                                    <div class="card-header bg-dark">
                                        <h1 class="display-6">` + name + `</h1>
                                        <button class="btn btn-outline-primary position-absolute top-0 end-0" style='width:80px;margin-right:105px;margin-top:15px' type="submit" onclick="labelModal('`+ name + `')">Label</button>
                                        <button class="btn btn-outline-success position-absolute top-0 end-0" style='width:80px;margin-right:15px;margin-top:15px' type="submit" onclick="makeTrainModal('`+ name + `')">Train</button>
                                    </div>
                            `


        if (datasets[n]['model'] != null) {
        html = `

                    <div class="row-auto"  style="padding:15px">
                    <div class="card shadow text-light bg-light" style="width: 700px; height:max-content;">
                        <div class="card-header bg-dark">
                        <h1 class="display-6">` + name + `</h1>
                        <button class="btn btn-outline-primary position-absolute top-0 end-0" style='width:80px;margin-right:195px;margin-top:15px' type="submit" onclick="labelModal('`+ name + `')">Label</button>
                        <button class="btn btn-outline-success position-absolute top-0 end-0" style='width:80px;margin-right:105px;margin-top:15px' type="submit" onclick="makeTrainModal('`+ name + `')">Train</button>
                        <button class="btn btn-outline-warning position-absolute top-0 end-0" style='width:80px;margin-right:15px;margin-top:15px' type="submit" onclick="showInferenceModal('`+ name + `')">Infer</button>
                        </div>
                    `
        }



        for (b = 0; b < behaviors.length; b++) {

        if (b == 0) {
            html += `
                        <div class="container" style='padding-top:10px'>
                            <div class="row">
                            <div class="col h5 text-dark">
                                Behavior
                            </div>
                            <div class="col h5 text-dark">
                                `+ mheaders[0] + `
                            </div>
                            <div class="col h5 text-dark">
                                `+ mheaders[1] + `
                            </div>
                            <div class="col h5 text-dark">
                                `+ mheaders[2] + `
                            </div>
                            <div class="col h5 text-dark">
                                `+ mheaders[3] + `
                            </div>
                            <div class="col h5 text-dark">
                                `+ mheaders[4] + `
                            </div>
                            </div>
                        </div>
                    `
        }

        html += `
                        <div class="container" style='padding-top:10px'>
                            <div class="row">
                            <div class="col h6 text-dark">
                                `+ behaviors[b] + `
                            </div>
                            <div class="col h6 text-dark" id="`+ names[i] + '-' + behaviors[b] + '-train' + `">
                                `+ metrics[behaviors[b]][mheaders[0]] + `
                            </div>
                            <div class="col h6 text-dark" id="`+ names[i] + '-' + behaviors[b] + '-test' + `">
                                `+ metrics[behaviors[b]][mheaders[1]] + `
                            </div>
                            <div class="col h6 text-dark" id="`+ names[i] + '-' + behaviors[b] + '-precision' + `">
                                `+ metrics[behaviors[b]][mheaders[2]] + `
                            </div>
                            <div class="col h6 text-dark" id="`+ names[i] + '-' + behaviors[b] + '-recall' + `">
                                `+ metrics[behaviors[b]][mheaders[3]] + `
                            </div>
                            <div class="col h6 text-dark" id="`+ names[i] + '-' + behaviors[b] + '-fscore' + `">
                                `+ metrics[behaviors[b]][mheaders[4]] + `
                            </div>
                            </div>
                        </div>
                    `
        }

        html += `
                    </div>
                </div>`

        container.innerHTML += html

    }
}

eel.expose(updateLabelImageSrc);
function updateLabelImageSrc(val) {
    let elem = document.getElementById('label-image');
    elem.src = "data:image/jpeg;base64, " + val
}

eel.expose(updateMetrics);
function updateMetrics(dataset, behavior, group, value) {
    let elem = document.getElementById(dataset + '-' + behavior + '-' + group);
    elem.innerHTML = `"` + value`"`
}

eel.expose(updateCount);
function updateCount(behavior, value) {
    let elem = document.getElementById('controls-' + behavior + '-count');
    if (elem != null) {
        elem.innerHTML = value
    }
}

async function trainModel() {
    let dataset = document.getElementById('tm-dataset').innerHTML
    let batchsize = document.getElementById('tm-batchsize').value
    let seqlen = document.getElementById('tm-seqlen').value
    let lrate = document.getElementById('tm-lrate').value
    let epochs = document.getElementById('tm-epochs').value

    await eel.train_model(dataset, batchsize, lrate, epochs, seqlen)()

    trainModal.hide()
    trainModal = new bootstrap.Modal(document.getElementById('trainModal'))
}

function makeTrainModal(name) {
    document.getElementById('tm-dataset').innerHTML = name

    trainModal.show()
}


async function labelModal(name) {
    let res = await eel.start_labeling(name)();
    if (res) {
        labeling = true

        let controls = document.getElementById('controls')

        behaviors = res[0]
        colors = res[1]

        controls.innerHTML += `
                        <div class="row">
                            <div class="col h6 text-light text-decoration-underline" style="max-width: 20vw">
                                Behavior
                            </div>
                            <div class="col d-flex justify-content-center align-items-center h6 text-light text-decoration-underline" style=" max-width:30px; margin-right:30px">
                                Code
                            </div>
                            <div class="col d-flex justify-content-center align-items-center h6 text-light text-decoration-underline" style="max-width:4vw">
                                Count
                            </div>
                        </div>`

        for (i = 0; i < behaviors.length; i++) {

        let k = i + 1

        if (i >= 9) {
            k = String.fromCharCode(97 + i - 9)
        }

        controls.innerHTML += `
                        <div class="row">
                            <div class="col h6 text-light" style="max-width: 20vw">
                                `+ behaviors[i] + `
                            </div>
                            <div class="col d-flex justify-content-center align-items-center h6" style="background-color:`+ colors[i] + `; max-width:30px; margin-right:30px">
                                `+ k + `
                            </div>
                            <div class="col d-flex justify-content-center align-items-center h6 text-light" id="controls-`+ behaviors[i] + '-' + 'count' + `" style="max-width:4vw">
                                0
                            </div>
                        </div>
                            `
        }


        document.getElementById('datasets').style.display = 'none'
        document.getElementById('label').style.display = 'flex'

    } else {
        document.getElementById('error-message').innerText = 'Error opening videos in this dataset.'
        let errorModal = new bootstrap.Modal(document.getElementById('errorModal'))
        errorModal.show()
    }

    /*
    setTimeout(function () {
    eel.update_counts()()
    }, 1000)
    */
}

async function createDataset() {
    let dirs = Object.keys(dir_children)

    if (dirs.length == 0) {
        document.getElementById('error-message').innerText = 'No recordings selected, try again.'
        let errorModal = new bootstrap.Modal(document.getElementById('errorModal'))
        errorModal.show()
        return
    }

    let recordings = []

    for (i = 0; i < dirs.length; i++) {
        let elem = document.getElementById(dirs[i])

        if (elem.checked) {
            recordings.push(dirs[i])
            continue
        }

        let subdirs = dir_children[dirs[i]]

        for (j = 0; j < subdirs.length; j++) {
            let elem = document.getElementById(dirs[i] + '-' + subdirs[j])


            if (elem.checked) {
            recordings.push(dirs[i] + '\\' + subdirs[j])
            }
        }
    }

    let name = document.getElementById('dataset-name').value
    let behaviors = document.getElementById('dataset-behaviors').value

    let behavior_array = behaviors.split(';')

    if (behavior_array.length <= 1 || behavior_array.length > 20) {
        document.getElementById('error-message').innerText = 'Must have between two to twenty behaviors specified, try again.'
        let errorModal = new bootstrap.Modal(document.getElementById('errorModal'))
        errorModal.show()
        return
    }

    let ba = []
    for (i = 0; i < behavior_array.length; i++) {
        if (behavior_array[i] == '') {
            continue;
        }
        ba.push(behavior_array[i].trim())
    }

    let ret = await eel.create_dataset(name, ba, recordings)();
    if (!ret) {
        document.getElementById('error-message').innerText = 'Unable to create the dataset.'
        let errorModal = new bootstrap.Modal(document.getElementById('errorModal'))
        errorModal.show()
        return
    }

    datasetModal.hide()
    datasetModal = new bootstrap.Modal(document.getElementById('addDataset'))

    loadInitial()
}

window.addEventListener("unload", function (e) {
    if (!routing) {
    eel.kill_streams();
    }
});

window.onbeforeunload = function () {
    if (!routing) {
    eel.kill_streams();
    }
}

function updateChildren(dir) {
    let subdirs = dir_children[dir]
    let parent = document.getElementById(dir)

    if (subdirs.length > 0) {
        for (i = 0; i < subdirs.length; i++) {
            let child = document.getElementById(dir + '-' + subdirs[i])
            child.checked = parent.checked
        }
    }
}

async function showDatasetModal() {
    let rt = await eel.get_record_tree()()

    if (rt) {
        let elem = document.getElementById('ad-recording-tree')

        dir_children = new Object()

        elem.innerHTML = ''

        let dirs = Object.keys(rt)

        for (i = 0; i < dirs.length; i++) {
            subdirs = rt[dirs[i]]

            dir_children[dirs[i]] = subdirs

            elem.innerHTML += `
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox" value="" id="`+ dirs[i] + `" onChange="updateChildren(` + dirs[i] + `)">
                        <label class="form-check-label" for="flexCheckDefault">
                        `+ dirs[i] + `
                        </label>
                    </div>
                    `
            for (j = 0; j < subdirs.length; j++) {
                elem.innerHTML += `
                        <div class="form-check" style='margin-left:10px'>
                        <input class="form-check-input" type="checkbox" value="" id="`+ dirs[i] + `-` + subdirs[j] + `">
                        <label class="form-check-label" for="flexCheckDefault">
                            `+ subdirs[j] + `
                        </label>
                        </div>
                        `
            }
        }

        datasetModal.show()
    } else {
        document.getElementById('error-message').innerText = 'No recordings yet, try this again when you have recorded videos.'
        let errorModal = new bootstrap.Modal(document.getElementById('errorModal'))
        errorModal.show()
    }
}

async function showInferenceModal(dataset) {

    document.getElementById('im-dataset').innerHTML = dataset

    let rt = await eel.get_record_tree()();

    if (rt) {
        let elem = document.getElementById('im-recording-tree')

        dir_children = new Object()

        elem.innerHTML = ''

        let dirs = Object.keys(rt)

        for (i = 0; i < dirs.length; i++) {
            subdirs = rt[dirs[i]]

            dir_children[dirs[i]] = subdirs

            elem.innerHTML += `
                <div class="form-check">
                    <input class="form-check-input" type="checkbox" value="" id="`+ dirs[i] + `" onChange="updateChildren(` + dirs[i] + `)">
                    <label class="form-check-label" for="flexCheckDefault">
                    `+ dirs[i] + `
                    </label>
                </div>
                `
            for (j = 0; j < subdirs.length; j++) {
                elem.innerHTML += `
                    <div class="form-check" style='margin-left:10px'>
                    <input class="form-check-input" type="checkbox" value="" id="`+ dirs[i] + `-` + subdirs[j] + `">
                    <label class="form-check-label" for="flexCheckDefault">
                        `+ subdirs[j] + `
                    </label>
                    </div>
                    `
            }

        }

        inferenceModal.show()
    } else {
        document.getElementById('error-message').innerText = 'No recordings yet, try this again when you have recorded videos.'
        let errorModal = new bootstrap.Modal(document.getElementById('errorModal'))
        errorModal.show()
    }
}

function startClassification() {
    dataset = document.getElementById('im-dataset').innerHTML

    let dirs = Object.keys(dir_children)

    if (dirs.length == 0) {
        document.getElementById('error-message').innerText = 'No recordings selected, try again.'
        let errorModal = new bootstrap.Modal(document.getElementById('errorModal'))
        errorModal.show()
        return
    }

    let recordings = []

    for (i = 0; i < dirs.length; i++) {
        let elem = document.getElementById(dirs[i])

        if (elem.checked) {
            recordings.push(dirs[i])
            continue
        }

        let subdirs = dir_children[dirs[i]]

        for (j = 0; j < subdirs.length; j++) {
            let elem = document.getElementById(dirs[i] + '-' + subdirs[j])

            if (elem.checked) {
                recordings.push(dirs[i] + '\\' + subdirs[j])
            }
        }
    }

    eel.start_classification(dataset, recordings)

    inferenceModal.hide()
    inferenceModal = new bootstrap.Modal(document.getElementById('inferenceModal'))
}


window.addEventListener("keydown", (event) => {
    if (labeling) {
    switch (event.key) {
        case "ArrowLeft":
            if (event.ctrlKey) {
                eel.next_video(-1);
            } else {
                eel.next_frame(-surf);
            }
            break;
        case "ArrowRight":
            if (event.ctrlKey) {
                eel.next_video(1);
            } else {
                eel.next_frame(surf);
            }
            break;
            case "ArrowUp":
            surf = surf * 2;
            break;
        case "ArrowDown":
            if (surf != 1) {
                surf = Math.trunc(surf / 2)
            }
            break;
        case "Delete":
            eel.delete_instance();
            break;

        case "Backspace":
            eel.pop_instance();
            break;
        default:
            if (event.keyCode >= 49 && event.keyCode <= 57) {
                eel.label_frame(event.keyCode - 49)
            } else if (event.keyCode >= 65 && event.keyCode <= 90) {
                eel.label_frame(event.keyCode - 65 + 9)
            }
            break;
    }

    }
})

window.setInterval(function () {
    loadInitial()
}, 10000)

window.setTimeout(function () {
    loadInitial()
}, 1000)