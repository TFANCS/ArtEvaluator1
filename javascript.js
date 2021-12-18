window.onload = function () {
    const spinner = document.getElementById('loading');
    spinner.classList.add('loaded');
}


const img = document.getElementById('file-preview');
const predictButton = document.getElementById("predict");
const scoreText = document.getElementById("art-score");
const calculatingText = document.getElementById("calculating-text");
const model = null;

document.getElementById('file-input').addEventListener('change', function (e) {
    var file = e.target.files[0];

    var blobUrl = window.URL.createObjectURL(file);

    img.src = blobUrl;
});




tf.loadGraphModel("https://tfancs.github.io/ArtEvaluator1/model/model.json").then(function(model) {
    window.model = model;
});
/*
tf.loadGraphModel("https://tfancs.github.io/ArtEvaluator1/model_archive/Model2Temp3/model.json").then(function(model) {
    this.model = model;
});
*/


// preprocess the image
function preprocessImage(image) {

    // resize the input image
    let tensor = tf.browser.fromPixels(image)
      .resizeNearestNeighbor([256, 256])
      .toFloat();
  
    // scale tensor image to range [0, 1] and add a dimension
    return tensor.div(255).expandDims();
}




async function buttonClickPredict(){


    input = preprocessImage(img);

    calculatingText.innerHTML = "Calculating..."
    scoreText.innerHTML = ""

    await sleep()

    window.model.predict(input).array().then(function(output){
        console.log(output);
        scoreText.innerHTML = Math.floor(output[0][0]*100);
        calculatingText.innerHTML = ""
    });

}


function sleep(){
    return new Promise(resolve => setTimeout(resolve, 1));
}

