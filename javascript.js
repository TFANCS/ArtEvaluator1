window.onload = function () {
    const spinner = document.getElementById('loading');
    spinner.classList.add('loaded');
}

document.getElementById('file-input').addEventListener('change', function (e) {
    var file = e.target.files[0];

    var blobUrl = window.URL.createObjectURL(file);

    var img = document.getElementById('file-preview');
    img.src = blobUrl;
});


const predictButton = document.getElementById("predict");
const scoreText = document.getElementById("art-score");

tf.loadLayersModel("model/model.json").then(function(model) {
    window.model = model;
});


window.model.predict([tf.tensor(input).reshape([1, 28, 28, 1])]).array().then(function(scores){
    scores = scores[0];
    scoreText.innerHTML = scores;
});


/*
let model;
async function loadModel() {

  // model name is "mobilenet"
  modelName = "mobilenet";
  
  model = undefined;
  
  // load the model using a HTTPS request (where you have stored your model files)
  model = await tf.loadLayersModel('https://gogul09.github.io/models/mobilenet/model.json');
  
  // hide model loading progress box
  loader.style.display = "none";
  load_button.disabled = true;
  load_button.innerHTML = "Loaded Model";
  console.log("model loaded..");
}

*/
