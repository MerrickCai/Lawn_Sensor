import processData from "./processData.js";
import { initImageDisplay, updateImages } from "./imageDisplay.js";
import uploadVideo from "./uploadVideo.js";

// Attach updateImages function to window to make it available in HTML
window.updateImages = updateImages;

window.addEventListener("DOMContentLoaded", async () => {
  try {
    // Initialize image display functionality
    initImageDisplay();

    // Initialize upload form functionality
    const uploadForm = document.querySelector("#panel-upload-data form");
    uploadForm.addEventListener("submit", uploadVideo);

    // Load JSON data
    const response = await fetch("./js/data/data.json");
    const data = await response.json();

    // Populate table, gallery, map, and charts panels
    processData(data);
  } catch (error) {
    console.error("Error loading JSON:", error);
  }
});
