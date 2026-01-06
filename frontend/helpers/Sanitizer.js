// Take Canvas input (left)
// Convert 420x420 to 28x28
// Then convert to Binary
// So it can be sent to the API

const blockSize = 15; // 420 / 28 = 15

// Input Processor

function getEMNISTInput(binaryArray, canvasWidth = 420, canvasHeight = 420, blockSize = 15) {
    const downsampled = new Float32Array(28 * 28);

    for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
            let sum = 0;
            for (let by = 0; by < blockSize; by++) {
                for (let bx = 0; bx < blockSize; bx++) {
                    const ix = x * blockSize + bx;
                    const iy = y * blockSize + by;
                    // Assuming binaryArray is 1 for White (bg), 0 for Black (ink)
                    sum += binaryArray[iy * canvasWidth + ix];
                }
            }
            
            // Calculate average brightness (0.0 to 1.0)
            let avg = sum / (blockSize * blockSize);

            // INVERT: Make background 0, Ink 1
            // IMPORTANT: PRESERVE THE GRAYSCALE!
            downsampled[y * 28 + x] = 1.0 - avg; 
        }
    }
    return Array.from(downsampled);
}
