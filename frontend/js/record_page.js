const ipc = window.ipcRenderer

let loaded = false

let routing = false

let cameraModal = new bootstrap.Modal(document.getElementById('addCamera'))
let statusModal = new bootstrap.Modal(document.getElementById('statusModal'))

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

async function loadCameraHTML() {
    let container = document.getElementById('camera-container');
    container.innerHTML = ""

    let cameras = await eel.get_camera_list()();

    for (const [cameraName, cameraSettings] of cameras) {
        let displayName = cameraName;

        /* We want to put an 'uncropped' note on any uncropped cameras. */
        let cx = parseFloat(cameraSettings['crop_left_x']);
        let cy = parseFloat(cameraSettings['crop_top_y']);
        let cw = parseFloat(cameraSettings['crop_width']);
        let ch = parseFloat(cameraSettings['crop_height']);

        let isCropped = cx != 0 || cy != 0 || cw != 1 || ch != 1
        if (!isCropped) {
            displayName += " (uncropped)"
        }

        // <img id="camera-` + cameraName + `" src="assets/noConnection.png" class="card-body d-flex justify-content-center align-items-center bg-light" style="width: 300px; height: 200px; padding: 0px; margin:0px"/>
        container.innerHTML += `
                    <div class="col-auto"">

                        <div class="card shadow text-white bg-dark mb-3" style="width: max-content; padding-left: 5px; padding-right: 5px; padding-bottom: 5px;">
                            <div class="card-header">
                                <h1 class="display-6">` + displayName + `</h1>
                            </div>
                    ` + `
                            <canvas id="camera-` + cameraName + `" width=300px height=300px></canvas>
                    ` + `
                            <div id="before-recording-` + cameraName + `" style="visibility: visible;">
                                <div class="btn rounded position-absolute bottom-0 end-0 bg-dark d-flex align-items-center justify-content-center" onclick="startCamera('` + cameraName + `')" style="margin-bottom:9px;margin-right:15px; width: 40px; height: 40px">
                                    <i class="bi bi-camera-video-fill" style="color:white; font-size:20px;"></i>
                                </div>
                                <div class="btn rounded position-absolute bottom-0 end-0 bg-dark d-flex align-items-center justify-content-center" onclick="loadCameraSettings('` + cameraName + `')" style="margin-bottom:9px;margin-right:60px; width: 40px; height: 40px">
                                    <i class="bi bi-crop" style="color:white; font-size:20px;"></i>
                                </div>
                                <div class="btn rounded position-absolute bottom-0 end-0 bg-dark d-flex align-items-center justify-content-center" onclick="liveViewCamera('` + cameraName + `')" style="margin-bottom:9px;margin-right:105px; width: 40px; height: 40px">
                                    <i class="bi bi-eye-fill" style="color:white; font-size:20px;"></i>
                                </div>
                            </div>
                            <div id="during-recording-` + cameraName + `" style="visibility: hidden;">
                                <div class="btn rounded position-absolute bottom-0 end-0 bg-dark d-flex align-items-center justify-content-center" onclick="stopCamera('` + cameraName + `','')" style="margin-bottom:9px;margin-right:15px; width: 40px; height: 40px">
                                    <i class="bi bi-square-fill" style="color:white; font-size:20px;"></i>
                                </div>
                                <div class="btn rounded position-absolute bottom-0 end-0 bg-dark d-flex align-items-center justify-content-center" onclick="liveViewCamera('` + cameraName + `')" style="margin-bottom:9px;margin-right:60px; width: 40px; height: 40px">
                                    <i class="bi bi-eye-fill" style="color:white; font-size:20px;"></i>
                                </div>
                            </div>
                        </div>
                    </div>`

        var canvas = document.getElementById('camera-' + cameraName);
        var ctx = canvas.getContext("2d");
        var image = new Image();
        image.src = "assets/noConnection.png";
        image.onload = function () {
            ctx.drawImage(image, 0, 0, 300, 300)
        }
    }
}

function setRecordAllIcon(isRecording) {
    const cameraFab = document.querySelector('.fab-container-right .fab:nth-child(2) i'); // Select the second button's icon

    if (isRecording) {
        // Change to "Stop" icon
        cameraFab.classList.remove('bi-camera-video-fill');
        cameraFab.classList.add('bi-square-fill');
        cameraFab.style.fontSize = '30px'; // Adjust size if needed
        cameraFab.onclick = stopAllCameras;
    } else {
        // Change to "Camera" icon
        cameraFab.classList.remove('bi-square-fill');
        cameraFab.classList.add('bi-camera-video-fill');
        cameraFab.style.fontSize = '25px'; // Adjust size if needed
        cameraFab.onclick = startAllCameras;
    }
}

eel.expose(updateImageSrc);
function updateImageSrc(name, val) {
    let canvas = document.getElementById('camera-' + name);
    var ctx = canvas.getContext("2d");

    var img = new Image();
    img.src = "data:image/jpeg;base64, " + val

    canvas.setAttribute("cbas_image_source", img.src)

    img.onload = async function () {
        const settings = await eel.get_camera_settings(name)();

        const cropX = settings['crop_left_x'] * img.width;
        const cropY = settings['crop_top_y'] * img.height;
        const cropWidth = settings['crop_width'] * img.width;
        const cropHeight = settings['crop_height'] * img.height;
        const resolution = settings['resolution']

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        console.log(cropX)
        console.log(cropY)
        console.log(cropWidth)
        console.log(cropHeight)
        drawImageScaled(img, ctx, cropX, cropY, cropWidth, cropHeight, resolution);
    }
}

async function updateCamButtons() {
    let cameraNames = (await eel.get_camera_list()()).map(([first, _]) => first);
    let activeNames = await eel.get_active_streams()();

    console.log(cameraNames)
    console.log(activeNames)

    if (activeNames) {
        const allActive = cameraNames.every(camera => activeNames.includes(camera));

        setRecordAllIcon(allActive)

        console.log(allActive)

        for (const activeName of activeNames) {

            let buttons = document.getElementById('before-recording-' + activeName)
            buttons.style.visibility = 'hidden';

            buttons = document.getElementById('during-recording-' + activeName)
            buttons.style.visibility = 'visible';
        }
    } else {
        setRecordAllIcon(false)
    }
}

async function loadCameras() {
    await loadCameraHTML();

    updateCamButtons();

    eel.download_camera_thumbnails()()
}

function updateCameras() {
    eel.download_camera_thumbnails()()

    updateCamButtons();
}

function drawImageScaled(img, ctx, sx, sy, sw, sh, resolution) {
    var canvas = ctx.canvas;
    var hRatio = canvas.width / sw;
    var vRatio = canvas.height / sh;
    var ratio = Math.min(hRatio, vRatio);
    var centerShift_x = (canvas.width - resolution) / 2;
    var centerShift_y = (canvas.height - resolution) / 2;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, sx, sy, sw, sh, centerShift_x, centerShift_y, resolution, resolution);
}

let oldCameraName = ""

async function loadCameraSettings(cameraName) {
    var canvas = document.getElementById("camera-image");
    var ctx = canvas.getContext("2d");

    let settings = await eel.get_camera_settings(cameraName)()
    if (settings) {

        url = settings['rtsp_url']
        framerate = settings['framerate']
        resolution = settings['resolution']
        cx = settings['crop_left_x']
        cy = settings['crop_top_y']
        cw = settings['crop_width']
        ch = settings['crop_height']

        oldCameraName = cameraName

        document.getElementById('cs-name').value = cameraName
        document.getElementById('cs-url').value = url
        document.getElementById('cs-framerate').value = framerate
        document.getElementById('cs-resolution').value = resolution
        document.getElementById('cs-cropx').value = cx
        document.getElementById('cs-cropy').value = cy
        document.getElementById('cs-crop-width').value = cw
        document.getElementById('cs-crop-height').value = ch

        document.getElementById('cs-resolution').onchange = drawBounds

        document.getElementById('cs-cropx').onchange = drawBounds
        document.getElementById('cs-cropy').onchange = drawBounds
        document.getElementById('cs-crop-width').onchange = drawBounds
        document.getElementById('cs-crop-height').onchange = drawBounds

        var image = new Image();
        image.src = document.getElementById('camera-' + cameraName).getAttribute("cbas_image_source")

        function drawBounds() {
            ctx.clearRect(0, 0, image.width, image.height)

            if (document.getElementById('cs-crop-width').value > 1 - document.getElementById('cs-cropx').value) {
                document.getElementById('cs-crop-width').value = 1 - document.getElementById('cs-cropx').value
            }

            if (document.getElementById('cs-crop-height').value > 1 - document.getElementById('cs-cropy').value) {
                document.getElementById('cs-crop-height').value = 1 - document.getElementById('cs-cropy').value
            }

            resolution = document.getElementById('cs-resolution').value

            x = document.getElementById('cs-cropx').value * image.width
            y = document.getElementById('cs-cropy').value * image.height
            w = document.getElementById('cs-crop-width').value * image.width
            h = document.getElementById('cs-crop-height').value * image.height

            // FUCK
            drawImageScaled(image, ctx, x, y, w, h, resolution);
        }

        image.onload = function () {
            drawBounds();
        };

        let modal = new bootstrap.Modal(document.getElementById('cameraSettings'))
        modal.show()
    }
}

async function saveCameraSettings() {
    let cameraName = document.getElementById('cs-name').value

    let cameraSettings = {
        "rtsp_url": document.getElementById('cs-url').value,
        "framerate": document.getElementById('cs-framerate').value,
        "resolution": document.getElementById('cs-resolution').value,
        'crop_left_x': document.getElementById('cs-cropx').value,
        'crop_top_y': document.getElementById('cs-cropy').value,
        'crop_width': document.getElementById('cs-crop-width').value,
        'crop_height': document.getElementById('cs-crop-height').value,
    }

    if (cameraName != oldCameraName) {
        await eel.rename_camera(oldCameraName, cameraName)
    }

    await eel.save_camera_settings(cameraName, cameraSettings)

    loadCameras()
}

function showCameraModal() {
    cameraModal.show()
}

async function showStatusModal() {
    let status = await eel.get_cbas_status()();
    let streams = status["streams"]
    if (!streams) {
        document.getElementById("status-streams").innerText = "No cameras are currently recording."
    } else {
        document.getElementById("status-streams").innerText = "Recording cameras: " + streams.join(", ")
    }

    let encodeFileCount = status["encode_file_count"]
    document.getElementById("status-encode-count").innerText = "Video files to encode: " + encodeFileCount;

    statusModal.show()
}

async function updateStatusModalIcon() {
    let cameraIcon = document.getElementById("status-camera-icon");
    let cameraOutline = document.getElementById("status-camera-outline");

    let status = await eel.get_cbas_status()();

    let streams = status["streams"]
    if (!streams) {
        cameraIcon.style.color = "white"; // Set to grey if inactive
        cameraIcon.style.animation = "none";

        cameraOutline.style.color = "white";
        cameraOutline.style.animation = "none";
    } else {
        cameraIcon.style.color = "red"; // Set to red if active
        cameraIcon.style.animation = "fadeInOut 1s infinite";

        cameraOutline.style.color = "red";
        cameraOutline.style.animation = "fadeInOut 1s infinite";
    }
}

async function addCamera() {
    let name = document.getElementById('camera-name').value
    let rtsp = document.getElementById('rtsp-url').value

    if (name == '' || rtsp == '') {
        document.getElementById('error-message').innerText = 'Please fill in all fields.'
        let errorModal = new bootstrap.Modal(document.getElementById('errorModal'))
        errorModal.show()
        return
    }

    let result = await eel.create_camera(name, rtsp)()

    if (result == null) {
        cameraModal.hide()
        cameraModal = new bootstrap.Modal(document.getElementById('addCamera'))
    } else {
        document.getElementById('error-message').innerText = ret
        let errorModal = new bootstrap.Modal(document.getElementById('errorModal'))
        errorModal.show()
    }

    loadCameras();
}

function removeCamera() {

}

function liveViewCamera(cameraName) {
    eel.open_camera_live_view(cameraName)
}

async function startCamera(cameraName) {
    let dir = await eel.create_recording_dir(cameraName)();
    if (!dir) {
        document.getElementById('error-message').innerText = 'Error, could not create a recording directory.'
        let errorModal = new bootstrap.Modal(document.getElementById('errorModal'))
        errorModal.show()
    }

    let res = await eel.start_camera_stream(cameraName, dir, 600)();
    if (res) {
        let buttons = document.getElementById('before-recording-' + cameraName)
        buttons.style.visibility = 'hidden';

        buttons = document.getElementById('during-recording-' + cameraName)
        buttons.style.visibility = 'visible';

    } else {
        document.getElementById('error-message').innerText = cameraName + ' is already being recorded.'
        let errorModal = new bootstrap.Modal(document.getElementById('errorModal'))
        errorModal.show()
    }

    updateStatusModalIcon();
}

async function stopCamera(cameraName) {
    let res = await eel.stop_camera_stream(cameraName)()

    if (res) {
        let buttons = document.getElementById('during-recording-' + cameraName)
        buttons.style.visibility = 'hidden';

        buttons = document.getElementById('before-recording-' + cameraName)
        buttons.style.visibility = 'visible';
    } else {
        document.getElementById('error-message').innerText = 'Error, could not stop recording.'
        let errorModal = new bootstrap.Modal(document.getElementById('errorModal'))
        errorModal.show()
    }

    updateStatusModalIcon();
}

async function startAllCameras() {
    let cameras = await eel.get_camera_list()();

    const promises = cameras.map(([cameraName, _]) => startCamera(cameraName));
    await Promise.all(promises);

    await updateCamButtons()
}

async function stopAllCameras() {
    let cameras = await eel.get_camera_list()();

    const promises = cameras.map(([cameraName, _]) => stopCamera(cameraName));
    await Promise.all(promises);

    await updateCamButtons()
}

loadCameras();

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

window.setInterval(function () {
    eel.get_progress_update()
}, 1000)

/* TODO: Rewrite the inference loading bar. */
eel.expose(inferLoadBar);
function inferLoadBar(progresses) {


    if (progresses) {

        let num = (1 / progresses.length) * 100
        let elem = document.getElementById('inference-bar');
        elem.style.visibility = 'visible'
        elem.innerHTML = ''

        for (i = 0; i < progresses.length; i++) {
            if (i < progresses.length - 1 && progresses[i] != 0) {
                if (progresses[i] < 0) {
                    elem.innerHTML += `<div class="progress-bar progress-bar-striped progress-bar-animated bg-danger" role="progressbar" style="width: ` + num + `%; border-right: 1px solid white" aria-valuenow="10" aria-valuemin="0" aria-valuemax="100"></div>`
                } else {
                    elem.innerHTML += `<div class="progress-bar progress-bar-striped progress-bar-animated bg-success" role="progressbar" style="width: ` + progresses[i] + `%; border-right: 1px solid white" aria-valuenow="10" aria-valuemin="0" aria-valuemax="100"></div>`
                }
            } else {
                if (progresses[i] < 0) {
                    elem.innerHTML += `<div class="progress-bar progress-bar-striped progress-bar-animated bg-danger" role="progressbar" style="width: ` + num + `%" aria-valuenow="10" aria-valuemin="0" aria-valuemax="100"></div>`
                } else {
                    elem.innerHTML += `<div class="progress-bar progress-bar-striped progress-bar-animated bg-success" role="progressbar" style="width: ` + progresses[i] + `%" aria-valuenow="10" aria-valuemin="0" aria-valuemax="100"></div>`
                }
            }
        }

    } else {

        window.setTimeout(function () {
            let elem = document.getElementById('inference-bar');
            elem.style.visibility = 'hidden'
        }, 1000)
    }

}