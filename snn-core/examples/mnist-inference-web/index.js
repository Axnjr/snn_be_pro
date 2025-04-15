/**
 * This demo supports both Burn and TensorFlow.js implementations
 * 
 * Released under a dual license: 
 * https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT
 * https://github.com/tracel-ai/burn/blob/main/LICENSE-APACHE
 */

/**
 * Auto crops the image, scales to 28x28 pixel image, and returns as grayscale image.
 * @param {object} mainContext - The 2d context of the source canvas.
 * @param {object} cropContext - The 2d context of an intermediate hidden canvas.
 * @param {object} scaledContext - The 2d context of the destination 28x28 canvas.
 */
export function cropScaleGetImageData(mainContext, cropContext, scaledContext) {

    const cropEl = cropContext.canvas;

    // Get the auto-cropped image data and put into the intermediate/hidden canvas
    cropContext.fillStyle = "rgba(255, 255, 255, 255)"; // white non-transparent color
    cropContext.fillRect(0, 0, cropEl.width, cropEl.height);
    cropContext.save();
    const [w, h, croppedImage] = cropImageFromCanvas(mainContext);
    cropEl.width = Math.max(w, h) * 1.2;
    cropEl.height = Math.max(w, h) * 1.2;
    const leftPadding = (cropEl.width - w) / 2;
    const topPadding = (cropEl.height - h) / 2;
    cropContext.putImageData(croppedImage, leftPadding, topPadding);

    // Copy image data to scale 28x28 canvas
    scaledContext.save();
    scaledContext.clearRect(0, 0, scaledContext.canvas.height, scaledContext.canvas.width);
    scaledContext.fillStyle = "rgba(255, 255, 255, 255)"; // white non-transparent color
    scaledContext.fillRect(0, 0, cropEl.width, cropEl.height);
    scaledContext.scale(28.0 / cropContext.canvas.width, 28.0 / cropContext.canvas.height);
    scaledContext.drawImage(cropEl, 0, 0);

    // Extract image data and convert into single value (greyscale) array
    const data = rgba2gray(scaledContext.getImageData(0, 0, 28, 28).data);
    scaledContext.restore();

    return data;
}

/**
 * Converts RGBA image data from canvas to grayscale (0 is white & 255 is black).
 * @param {int[]} - Image data.
 */
export function rgba2gray(data) {
    let converted = new Float32Array(data.length / 4);

    // Data is stored as [r0,g0,b0,a0, ... r[n],g[n],b[n],a[n]] where n is number of pixels.
    for (let i = 0; i < data.length; i += 4) {
        let r = 255 - data[i];     // red
        let g = 255 - data[i + 1]; // green
        let b = 255 - data[i + 2]; // blue
        let a = 255 - data[i + 3]; // alpha

        // Use RGB grayscale coefficients (https://imagej.nih.gov/ij/docs/menus/image.html)
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        converted[i / 4] = y; // 4 times fewer data points but the same number of pixels.
    }
    return converted;
}

/**
 * Auto crops a canvas images and returns its image data.
 * @param {object} ctx - canvas 2d context.
 * src: https://stackoverflow.com/a/22267731
 */
export function cropImageFromCanvas(ctx) {
    let canvas = ctx.canvas,
        w = canvas.width,
        h = canvas.height,
        pix = { x: [], y: [] },
        imageData = ctx.getImageData(0, 0, canvas.width, canvas.height),
        x,
        y,
        index;
    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            index = (y * w + x) * 4;

            let r = imageData.data[index];
            let g = imageData.data[index + 1];
            let b = imageData.data[index + 2];
            if (Math.min(r, g, b) != 255) {
                pix.x.push(x);
                pix.y.push(y);
            }
        }
    }
    pix.x.sort(function (a, b) {
        return a - b;
    });
    pix.y.sort(function (a, b) {
        return a - b;
    });
    let n = pix.x.length - 1;
    w = 1 + pix.x[n] - pix.x[0];
    h = 1 + pix.y[n] - pix.y[0];
    return [w, h, ctx.getImageData(pix.x[0], pix.y[0], w, h, { willReadFrequently: true })];
}

/**
 * Truncates number to a given decimal position
 * @param {number} num - Number to truncate.
 * @param {number} fixed - Decimal positions.
 * src: https://stackoverflow.com/a/11818658
 */
export function toFixed(num, fixed) {
    const re = new RegExp('^-?\\d+(?:\.\\d{0,' + (fixed || -1) + '})?');
    return num.toString().match(re)[0];
}

/**
 * Looks up element by an id.
 * @param {string} - Element id.
 */
export function $(id) {
    return document.getElementById(id);
}

/**
 * Helper function that builds a chart using Chart.js library.
 * @param {object} chartEl - Chart canvas element.
 * 
 * NOTE: Assumes chart.js is loaded into the global.
 */
export function chartConfigBuilder(chartEl) {
    Chart.register(ChartDataLabels);
    return new Chart(chartEl, {
        plugins: [ChartDataLabels],
        type: "bar",
        data: {
            labels: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            datasets: [
                {
                    data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    borderWidth: 0,
                    fill: true,
                    backgroundColor: "#247ABF",
                },
            ],
        },
        options: {
            responsive: false,
            maintainAspectRatio: false,
            animation: true,
            plugins: {
                legend: {
                    display: false,
                },
                tooltip: {
                    enabled: true,
                },
                datalabels: {
                    color: "white",
                    formatter: function (value, context) {
                        return toFixed(value, 2);
                    },
                },
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                },
            },
        },
    });
}

/**
 * TensorFlow.js implementation of MNIST inference
 */
class TensorflowMnist {
    constructor() {
        this.model = null;
        this.loadModel();
    }

    async loadModel() {
        // Load the TensorFlow.js model
        this.model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mnist_v1/model.json');
    }

    async inference(imageData) {
        // Prepare the input tensor
        const tensor = tf.tensor(imageData).reshape([1, 28, 28, 1]);
        let predictions;
        alert("inference");
        for (let i = 0; i < 100; i++) {
            await Promise.resolve(3000);
        }
        alert("inference2");
        predictions = await this.model.predict(tensor).data();
        return Array.from(predictions);
    }
}

/**
 * Factory class to manage different MNIST implementations
 */
export class MnistFactory {
    static async create(version) {
        if (version === 'tensorflow') {
            return new TensorflowMnist();
        } else {
            const wasm = await import("./pkg/mnist_inference_web.js");
            await wasm.default();
            return new wasm.Mnist();
        }
    }
}

/**
 * Initialize the MNIST demo with the selected implementation
 */
export async function initMnistDemo(version, fabricCanvas, mainContext, cropContext, scaledContext, chart, dur) {
    // alert("initMnistDemo" + version + " " + dur);
    const mnist = await MnistFactory.create(version);
    let timeoutId;
    let isDrawing = false;
    let isTimeOutSet = false;

    async function fireOffInference() {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(async () => {
            isTimeOutSet = true;
            fabricCanvas.freeDrawingBrush._finalizeAndAddPath();
            const data = cropScaleGetImageData(mainContext, cropContext, scaledContext);
            const output = await mnist.inference(data);
            chart.data.datasets[0].data = output;
            chart.update();
            isTimeOutSet = false;
        }, dur);
        isTimeOutSet = true;
    }

    fabricCanvas.on("mouse:down", function (event) {
        isDrawing = true;
    });

    fabricCanvas.on("mouse:up", async function (event) {
        isDrawing = false;
        await fireOffInference();
    });

    fabricCanvas.on("mouse:move", async function (event) {
        if (isDrawing && isTimeOutSet == false) {
            await fireOffInference();
        }
    });

    return {
        fireOffInference
    };
}