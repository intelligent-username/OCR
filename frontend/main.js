// main.js

const canvas = document.getElementById("draw-canvas");
// Use willReadFrequently to optimize repeated getImageData readbacks
const ctx = canvas.getContext("2d", { willReadFrequently: true });

const predCanvas = document.getElementById("prediction-canvas");
const predCtx = predCanvas.getContext("2d");

// Canvas drawing setup
ctx.lineWidth = 42;
ctx.lineCap = "round";
ctx.strokeStyle = "black";

// Track drawing state
let drawing = false;
let erase = false;
let lastLogTime = 0;
// Milliseconds between read/process cycles (tweakable)
let logGapMs = 250;
// Processing loop control
let processingLoopRunning = false;

// Prevent context menu on right click
canvas.addEventListener("contextmenu", (e) => {
    e.preventDefault();
});

canvas.addEventListener("mousedown", (e) => {
    drawing = true;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    ctx.beginPath();
    ctx.moveTo(x, y);
    
    if (e.button === 2) { // Right click
        erase = true;
        ctx.strokeStyle = "white";
    } else {
        erase = false;
        ctx.strokeStyle = "black";
    }

    // Start the processing loop if not already running
    if (!processingLoopRunning) {
        processingLoopRunning = true;
        requestAnimationFrame(processingLoop);
    }
});

canvas.addEventListener("mouseup", () => {
    drawing = false;
    erase = false;
    ctx.strokeStyle = "black";
});

// Processing loop runs while `drawing` is true and throttles work via `logGapMs`.
function processingLoop() {
    if (!drawing) {
        processingLoopRunning = false;
        return;
    }

    const now = Date.now();
    if (now - lastLogTime >= logGapMs) {
        lastLogTime = now;

        // Schedule after paint
        requestAnimationFrame(() => {
            try {
                // 1. Extract and Normalize (Don't Binarize!)
                const inputArray = ctx.getImageData(0, 0, canvas.width, canvas.height);
                const alphaValues = Array.from(inputArray.data)
                    .filter((_, i) => i % 4 === 3) // Keep only Alpha
                    .map(a => a / 255.0);          // Normalize 0-255 to 0.0-1.0

                // 2. Downsample (Using the modified function below)
                const temp = getEMNISTInput(alphaValues);

                // console.log("Input Array: ", inputArray);
                // console.log("Temp: ", temp);

                // 3. Visualize Prediction Input (Grayscale support)
                predCtx.clearRect(0, 0, predCanvas.width, predCanvas.height);
                const scale = predCanvas.width / 28;

                for (let yy = 0; yy < 28; yy++) {
                    for (let xx = 0; xx < 28; xx++) {
                        const val = temp[yy * 28 + xx];
                        
                        // VISUALIZATION ONLY: 
                        // We want Ink (1.0) to look Black, and Bg (0.0) to look White.
                        // So we invert the color calculation for the human eye.
                        const c = Math.floor(255 * (1 - val)); 
                        predCtx.fillStyle = `rgb(${c}, ${c}, ${c})`;
                        predCtx.fillRect(xx * scale, yy * scale, scale, scale);
                    }
                }

                // Send to server with error handling (non-blocking)
                fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ image: temp }),
                })
                .then(async response => {
                    if (!response.ok) {
                        const text = await response.text();
                        throw new Error(`Server error: ${response.status} ${response.statusText} - ${text}`);
                    }
                    return response.json();
                })
                .then(data => {
                    console.log("Predictions:", data.predictions);
                })
                .catch(error => console.error("Error:", error));
            } catch (err) {
                console.error("Processing loop error:", err);
            }
        });
    }

    requestAnimationFrame(processingLoop);
}
canvas.addEventListener("mouseleave", () => {
    drawing = false;
    erase = false;
    ctx.strokeStyle = "black";
});
canvas.addEventListener("mousemove", draw);

canvas.addEventListener("touchstart", (e) => {
    e.preventDefault();
    drawing = true;
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    ctx.beginPath();
    ctx.moveTo(touch.clientX - rect.left, touch.clientY - rect.top);

    // Start the processing loop if not already running
    if (!processingLoopRunning) {
        processingLoopRunning = true;
        requestAnimationFrame(processingLoop);
    }
}, { passive: false });

canvas.addEventListener("touchmove", (e) => {
    e.preventDefault();
    if (!drawing) return;
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    ctx.lineTo(touch.clientX - rect.left, touch.clientY - rect.top);
    ctx.stroke();
}, { passive: false });

canvas.addEventListener("touchend", (e) => {
    drawing = false;
}, { passive: false });

// Draw function for mouse
function draw(e) {
    if (!drawing) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    ctx.lineTo(x, y);
    ctx.stroke();
    
    // (Processing loop now handles getImageData and updates; draw() stays lightweight)
}

// Clear button
document.getElementById("clear-btn").addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Clear and reset prediction canvas to white so it's visually clear
    predCtx.clearRect(0, 0, predCanvas.width, predCanvas.height);
    predCtx.fillStyle = 'white';
    predCtx.fillRect(0, 0, predCanvas.width, predCanvas.height);
});
