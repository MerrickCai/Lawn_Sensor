import processData from "./processData.js";

window.addEventListener("DOMContentLoaded", async () => {
  try {
    const response = await fetch("data.json");
    const data = await response.json();

    // Populate table, gallery, map, and charts panels
    processData(data);
  } catch (error) {
    console.error("Error loading JSON:", error);
  }
});
