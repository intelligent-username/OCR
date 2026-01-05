const slider = document.getElementById("k-slider");
const output = document.getElementById("k-value");

// Initialize w/ default
output.textContent = slider.value;

// Update number instantly on drag
slider.addEventListener("input", function() {
    output.textContent = this.value;
});
