
class InvertedResidualBlock extends tf.layers.Layer{
    constructor(config){
        super(config);

        this.kernelSize = config.kernelSize
        this.strides = config.strides
        this.filters = config.filters

        this.conv2a = tf.layers.conv2d({filters : this.filters, kernelSize : [1, 1], strides : 1, padding : 'same'})
        this.bn2a = tf.layers.batchNormalization({})
        this.dconv2a = tf.layers.depthwiseConv2d({kernelSize : this.kernelSize, strides : this.strides, padding : 'same'})
        this.bn2b = tf.layers.batchNormalization({})
        this.conv2b = tf.layers.conv2d({filters : this.filters, kernelSize : [1, 1], strides : 1, padding : 'same'})
        this.bn2c = tf.layers.batchNormalization({})
    }

    build(inputShape){
        this.same_output_size = ((this.strides == 1) && (this.filters == inputShape[3]))
        if (!this.same_output_size){
            this.conv2_sc = tf.layers.conv2d({filters : this.filters, kernelSize : [1, 1], strides : this.strides, padding : 'same'})
        }
    }

    getConfig() {
        const config = super.getConfig();
        Object.assign(config, {
            'kernelSize' : this.kernelSize,
            'strides' : this.strides,
            'filters' : this.filters
        });
        return config;
    }

    call(inputs, training=None){

        x = this.conv2a(inputs, training=training)
        x = this.bn2a(x)
        x1 = tf.nn.relu(x)
        x = tf.nn.relu(x)

        x = this.dconv2a(x)
        x = this.bn2b(x)
        x = tf.nn.relu(x)

        x = this.conv2b(x)
        x = this.bn2c(x)
        x = tf.nn.relu(x)

        if (this.same_output_size){
            x += inputs
        }else{
            x1 = this.conv2_sc(x1)
            x += x1
        }
        return x
    }

    static get className() {
        return "InvertedResidualBlock";
    }
}


class SqueezeAndExcitationBlock extends tf.layers.Layer{
    constructor(config){
        super(config);

        this.channels = config.channels
        this.r = config.r

        if(this.r == null){
            this.r = 8
        }

        this.gap = tf.layers.globalAveragePooling2d({})

        this.dencea = tf.layers.dense({units : Math.floor(this.channels/this.r), activation : "relu"})
        this.denseb = tf.layers.dense({units : this.channels, activation : "sigmoid"})

        this.multiply = tf.layers.multiply({})
    }

    getConfig(){
        const config = super.getConfig();
        Object.assign(config, {
            'channels' : this.channels,
            'r' : this.r,
        });
        return config;
    }

    call(inputs, training=None){

        x = this.gap(inputs)
        x = this.dencea(x)
        x = this.denseb(x)
        x = this.multiply([inputs, x])

        return x
    }

    static get className() {
        return "SqueezeAndExcitationBlock";
    }
}



class HardSwish extends tf.layers.Layer{
    constructor(config){
        super(config);
    }

    getConfig(){
        const config = super.getConfig();
        Object.assign(config, {
            'channels' : this.channels,
            'r' : this.r,
        });
        return config;
    }

    call(inputs){
        return inputs * (tf.nn.relu6(inputs + 3) / 6)
    }

    compute_output_shape(inputShape){
        return inputShape
    }

    static get className() {
        return "HardSwish";
    }
}



class InvertedResidualBlockWithSE extends tf.layers.Layer{
    constructor(config){
        super(config);

        this.kernelSize = config.kernelSize
        this.strides = config.strides
        this.filters = config.filters

        this.conv2a = tf.layers.conv2d({filters : this.filters, kernelSize : [1, 1], strides : 1, padding : 'same'})
        this.bn2a = tf.layers.batchNormalization({})
        this.dconv2a = tf.layers.depthwiseConv2d({kernelSize : this.kernelSize, strides : this.strides, padding : 'same'})
        this.bn2b = tf.layers.batchNormalization({})

        this.se = new SqueezeAndExcitationBlock({channels : this.filters})

        this.conv2b = tf.layers.conv2d({filters : this.filters, kernelSize : [1, 1], strides : 1, padding : 'same'})
        this.bn2c = tf.layers.batchNormalization({})
    }


    build(inputShape){
        this.same_output_size = ((this.strides == 1) && (this.filters == inputShape[3]))
        if (!this.same_output_size){
            this.conv2_sc = tf.layers.conv2d({filters : this.filters, kernelSize : [1, 1], strides : this.strides, padding : 'same'})
        }
    }


    getConfig(){
        const config = super.getConfig();
        config = {
            'kernelSize' : this.kernelSize,
            'strides' : this.strides,
            'filters' : this.filters
        }
        return config;
    }


    call(inputs, training=None){

        x = this.conv2a(inputs)
        x = this.bn2a(x)
        x1 = tf.nn.relu(x)
        x = tf.nn.relu(x)

        x = this.dconv2a(x)
        x = this.bn2b(x)
        x = tf.nn.relu(x)

        x = this.se(x)

        x = this.conv2b(x)
        x = this.bn2c(x)
        x = tf.nn.relu(x)

        if(this.same_output_size){
            x += inputs
        }else{
            x1 = this.conv2_sc(x1)
            x += x1
        }
        return x
    }

    static get className() {
        return "InvertedResidualBlockWithSE";
    }
}


tf.serialization.registerClass(HardSwish);
tf.serialization.registerClass(SqueezeAndExcitationBlock);
tf.serialization.registerClass(InvertedResidualBlock);
tf.serialization.registerClass(InvertedResidualBlockWithSE);


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




tf.loadLayersModel("https://tfancs.github.io/ArtEvaluator1/model/model.json").then(function(model) {
    this.model = model;
});




// preprocess the image
function preprocessImage(image) {

    // resize the input image
    let tensor = tf.browser.fromPixels(image)
      .resizeNearestNeighbor([256, 256])
      .toFloat();
  
    // scale tensor image to range [0, 1] and add a dimension
    return tensor.div(255).expandDims();
}




function buttonClickPredict(){

    input = preprocessImage(img);

    console.log(input)

    calculatingText.innerHTML = "Calculating..."

    window.model.predict(input).array().then(function(output){
        console.log(output)
        scoreText.innerHTML = Math.floor(output[0][0]*100);
    });

    calculatingText.innerHTML = ""
}

