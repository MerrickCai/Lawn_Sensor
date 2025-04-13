import processData from "./processData.js";
import { initImageDisplay, updateImages } from "./imageDisplay.js";

// Attach updateImages function to window to make it available in HTML
window.updateImages = updateImages;

window.addEventListener("DOMContentLoaded", async () => {
  try {
    // Initialize image display functionality
    initImageDisplay();

    const response = await fetch("./js/data.json");
    const data = await response.json();

    // Populate table, gallery, map, and charts panels
    processData(data);
  } catch (error) {
    console.error("Error loading JSON:", error);
  }
});
