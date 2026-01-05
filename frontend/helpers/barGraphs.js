function drawPredictionGraph(canvas, predictions) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    const padding = 20;
    const labelSpace = 40; // Space for character labels on left
    const barSpace = w - padding * 2 - labelSpace - 60; // Max width for bars (leaving room for %)
    
    // Clear and set white background
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, w, h);

    const barHeight = (h - 2 * padding) / predictions.length - 10;

    predictions.forEach((pred, i) => {
        const y = padding + i * (barHeight + 10);
        const barWidth = Math.max(pred.prob * barSpace, 2); // Ensure tiny bars are visible

        // 1. Draw Label (Character)
        ctx.fillStyle = "black";
        ctx.font = "bold 24px monospace";
        ctx.textAlign = "center";
        ctx.fillText(pred.char, padding + labelSpace / 2, y + barHeight / 1.5);

        // 2. Draw Bar
        ctx.fillStyle = pred.prob > 0.5 ? "black" : "#555"; // Darker for high confidence
        ctx.fillRect(padding + labelSpace, y, barWidth, barHeight);

        // 3. Draw Percentage
        ctx.fillStyle = "#333";
        ctx.font = "14px monospace";
        ctx.textAlign = "left";
        ctx.fillText((pred.prob * 100).toFixed(2) + "%", padding + labelSpace + barWidth + 5, y + barHeight / 1.6);
    });
}
