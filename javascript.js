window.onload = function () {
    const spinner = document.getElementById('loading');
    spinner.classList.add('loaded');
}


const img = document.getElementById('file-preview');
const predictButton = document.getElementById("predict");
const scoreText = document.getElementById("art-score");


document.getElementById('file-input').addEventListener('change', function (e) {
    var file = e.target.files[0];

    var blobUrl = window.URL.createObjectURL(file);

    img.src = blobUrl;
});




tf.loadLayersModel("https://tfancs.github.io/ArtEvaluator1/model/model.json").then(function(model) {
    window.model = model;
});


// preprocess the image
function preprocessImage(image) {

    // resize the input image
    let tensor = tf.browser.fromPixels(image)
      .resizeNearestNeighbor([256, 256])
      .toFloat();
  
    // scale tensor image to range [0, 1]
    return tensor.div(255).expandDims();
}




function buttonClickPredict(){

    input = preprocessImage(img);

    console.log(input)

    window.model.predict(input).array().then(function(output){
        console.log(output)
        scoreText.innerHTML = Math.floor(output[0][0]*100);
    });
}

