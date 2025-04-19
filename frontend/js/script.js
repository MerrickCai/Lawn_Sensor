import processData from "./processData.js";
import uploadVideo from "./upload/uploadVideo.js";
import initImageDisplay from "./upload/initImageDisplay.js";
import updateImages from "./upload/updateImages.js";
import GenerateImages from "./upload/GenerateImages.js";
import loadGPS from "./upload/loadGPS.js";
// Attach updateImages function to window to make it available in HTML
window.updateImages = updateImages;

window.addEventListener("DOMContentLoaded", async () => {
  try {
    // Initialize image display functionality
    initImageDisplay();

    // Initialize upload form functionality
    const uploadForm = document.querySelector("#panel-upload-data form");
    uploadForm.addEventListener("submit", uploadVideo);

    // Setup button event listeners
    const generateImagesBtn = document.getElementById("generateImagesBtn");
    generateImagesBtn.addEventListener("click", GenerateImages);

    const loadGPSBtn = document.getElementById("loadGPSBtn");
    loadGPSBtn.addEventListener("click", loadGPS);
    // Load JSON data
    const response = await fetch("./js/data/data.json");
    const data = await response.json();

    // Populate table, gallery, map, and charts panels
    processData(data);
  } catch (error) {
    console.error("Error loading JSON:", error);
  }
});

