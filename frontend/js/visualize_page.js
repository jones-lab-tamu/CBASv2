const ipc = window.ipcRenderer

let loaded = false
let routing = false

let vis_root_dir = null 
let vis_sub_dir = null 
let vis_model = null 
let vis_behavior = null

let dir_children = new Object();

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

window.addEventListener("unload", function(e){
    if(!routing) {
        eel.kill_streams();
    }
});

window.onbeforeunload = function (){
    if(!routing) {
        eel.kill_streams();
    }
}

eel.expose(updateActogram);
function updateActogram(val) {
    // showProgressBar(); // 3.4, showing progress bar, 3.5 commented until load details are right
    showBuffering(); //3.5, buffering signal load

    let elem = document.getElementById('actogram-image');
    elem.src = "data:image/jpeg;base64, " + val

    elem.style.display="block"

    hideBuffering(); //3.5, buffering signal hide

}

// added 3.5, in place of load bar, buffering funcs, for now...
function showBuffering() {
    let spinner = document.getElementById('loading-spinner');
    spinner.style.display = "block";
}
function hideBuffering() {
    let spinner = document.getElementById('loading-spinner');
    spinner.style.display = "none";
}

function toggleVis(id) {
    let elem = document.getElementById(id)

    if(elem.style.display=='none') {
        elem.style.display = 'block'
    } else {
        elem.style.display = 'none'
    }

}

function setValues(rd, sd, md, beh) {
    let fr = document.getElementById('vs-framerate').value
    let bs = document.getElementById('vs-binsize').value
    let st = document.getElementById('vs-start').value
    let col = document.getElementById('vs-color').value
    let th = document.getElementById('vs-threshold').value
    let norm = document.getElementById('vs-norm').value
    let lc = document.getElementById('vs-lcycle').value

    let titleElem = document.getElementById('actogram-title') // added 2/23, acto title
    titleElem.textContent = "Actogram for " + md + " - " + beh // added 2/23 ^^

    showBuffering(); // added 3.5, buffering siganl
    eel.make_actogram(rd, sd, md, beh, fr, bs, st, col, th, norm, lc)
}

function adjustActogram() {
    let fr = document.getElementById('vs-framerate').value
    let bs = document.getElementById('vs-binsize').value
    let st = document.getElementById('vs-start').value
    let col = document.getElementById('vs-color').value
    let th = document.getElementById('vs-threshold').value
    let norm = document.getElementById('vs-norm').value
    let lc = document.getElementById('vs-lcycle').value

    eel.adjust_actogram(fr, bs, st, col, th, norm, lc)
}

async function initialize() {
    let directories = document.getElementById('directories')

    let recording_tree = await eel.get_recording_tree()()
    if (!recording_tree) {
        document.getElementById('error-message').innerText = 'No recordings yet, try this again when you have recorded videos.'
        let errorModal = new bootstrap.Modal(document.getElementById('errorModal'))
        errorModal.show()
        return;
    }

    html = ''

    for(i = 0; i < recording_tree.length; i++) {
        let date = recording_tree[i]
        let date_str = recording_tree[i][0]

        html += `
                <h3 class='text-light' onclick="toggleVis('rd` + date_str + `')" style="cursor:pointer;">` + date_str +`
                </h3>`

        html +=`
                <div id='rd` + date_str + `' style=" display:none;">`

        for (j = 0; j < date[1].length; j++) {
            let time = date[1][j]
            let time_str = time[0]

            html += `
                    <h4 class='text-light' onclick="toggleVis('rd`+ date_str +`sd`+ time_str +`')" style="cursor:pointer;padding-left:10px">`+ time_str +`
                    </h4>`

            html +=`
                    <div id='rd` + date_str + `sd` + time_str + `' style=" display:none;">`

            for (k = 0; k < time[1].length; k++) {
                let model = time[1][k];
                let model_str = model[0]

                html += `
                        <h5 class='text-light' onclick="toggleVis('rd`+ date_str +`sd` + time_str + `md` + model_str + `')" style="cursor:pointer;padding-left:20px">` + model_str + `
                        </h5>`

                html +=`
                        <div id='rd` + date_str + `sd` + time_str + `md` + model_str + `' style="display:none;padding-left:30px;">`
                
                for(b = 0; b < model[1].length; b++) {
                    let behavior_str = model[1][b];

                    html += `
                    <div class="form-check">
                        <input class="form-check-input" type="radio" name="flexRadioDefault" id="md`+model_str +`-`+behavior_str+`" onclick="setValues('`+date_str+`','`+time_str+`','`+model_str+`','`+behavior_str+`')">
                        <label class="form-check-label text-light h6" for="md`+model_str+`-`+behavior_str+`">
                            `+behavior_str+`
                        </label>
                    </div>`

                }

                html +=`
                </div>
                    `

            }

            html +=`
            </div>
                `

        }

        html +=`
        </div>
            `
    }

    directories.innerHTML = html
}


setTimeout(function (){
    initialize()
}, 500)

setTimeout(function (){
    setInterval(function(){
        adjustActogram()
    }, 1000)
}, 500)
