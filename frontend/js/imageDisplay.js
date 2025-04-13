/**
 * Initialize image display functionality
 */
export function initImageDisplay() {
  // Initially hide all images
  const images = document.querySelectorAll(".image");
  images.forEach((img) => {
    img.style.display = "none";
  });

  // Add default event handling for checkboxes
  const checkboxes = document.querySelectorAll('.checkbox-label input[type="checkbox"]');
  checkboxes.forEach((checkbox) => {
    checkbox.checked = false;
  });

  // Set initial z-index values for proper layering
  const image1 = document.getElementById("image1");
  const image2 = document.getElementById("image2");
  const image3 = document.getElementById("image3");

  if (image1) image1.style.zIndex = "1";
  if (image2) image2.style.zIndex = "2";
  if (image3) image3.style.zIndex = "3";

  // Set z-index for plant images
  for (let i = 1; i <= 13; i++) {
    const plantImage = document.getElementById(`plantImage${i}`);
    if (plantImage) {
      plantImage.style.zIndex = `${i + 3}`;
    }
  }

  // Setup button event listeners if they exist
  const generateFramesBtn = document.getElementById("generateFramesBtn");
  const generateImagesBtn = document.getElementById("generateImagesBtn");

  if (generateFramesBtn) {
    generateFramesBtn.addEventListener("click", generateFrames);
  }

  if (generateImagesBtn) {
    generateImagesBtn.addEventListener("click", runTest);
  }
}

/**
 * Update image display status based on checkbox controls
 */
export function updateImages() {
  // Get main image checkboxes
  const image1Checkbox = document.getElementById("image1Checkbox");
  const image2Checkbox = document.getElementById("image2Checkbox");
  const image3Checkbox = document.getElementById("image3Checkbox");

  // Get main images
  const image1 = document.getElementById("image1");
  const image2 = document.getElementById("image2");
  const image3 = document.getElementById("image3");

  // Reset display for main images
  if (image1) image1.style.display = "none";
  if (image2) image2.style.display = "none";
  if (image3) image3.style.display = "none";

  // Reset display for plant images
  for (let i = 1; i <= 13; i++) {
    const plantImage = document.getElementById(`plantImage${i}`);
    if (plantImage) {
      plantImage.style.display = "none";
    }
  }

  let numberOfImagesDisplayed = 0;

  // Process main image checkboxes
  if (image1Checkbox && image1Checkbox.checked && image1) {
    image1.style.display = "block";
    numberOfImagesDisplayed++;
  }

  if (image2Checkbox && image2Checkbox.checked && image2) {
    image2.style.display = "block";
    numberOfImagesDisplayed++;
  }

  if (image3Checkbox && image3Checkbox.checked && image3) {
    image3.style.display = "block";
    numberOfImagesDisplayed++;
  }

  // Process plant image checkboxes
  let displayedPlantImages = 0;
  for (let i = 1; i <= 13; i++) {
    const plantCheckbox = document.getElementById(`plant${i}Checkbox`);
    const plantImage = document.getElementById(`plantImage${i}`);

    if (plantCheckbox && plantCheckbox.checked && plantImage) {
      plantImage.style.display = "block";
      displayedPlantImages++;
    }
  }

  // Adjust main images opacity based on number displayed
  if (numberOfImagesDisplayed > 1) {
    const opacity = 1 / numberOfImagesDisplayed;
    if (image1 && image1.style.display === "block") {
      image1.style.opacity = opacity;
    }
    if (image2 && image2.style.display === "block") {
      image2.style.opacity = opacity;
    }
    if (image3 && image3.style.display === "block") {
      image3.style.opacity = opacity;
    }
  } else {
    if (image1) image1.style.opacity = "1";
    if (image2) image2.style.opacity = "1";
    if (image3) image3.style.opacity = "1";
  }

  // Adjust plant images opacity if they're displayed
  if (displayedPlantImages > 0) {
    const plantOpacity = displayedPlantImages > 1 ? 0.333 : 1;
    for (let i = 1; i <= 13; i++) {
      const plantImage = document.getElementById(`plantImage${i}`);
      if (plantImage && plantImage.style.display === "block") {
        plantImage.style.opacity = plantOpacity;
      }
    }
  }

  // Log plant filtering status for debugging/development
  const plantFilterStatus = {};
  for (let i = 1; i <= 13; i++) {
    const checkbox = document.getElementById(`plant${i}Checkbox`);
    if (checkbox) {
      const plantNames = [
        "Blue Violets",
        "Broadleaf Plantains",
        "Common Ivy",
        "Common Purslane",
        "Eastern Poison Ivy",
        "Japanese Honeysuckle",
        "Oxeye Daisy",
        "Roundleaf greenbrier",
        "Virginia Creeper",
        "Wild Garlic",
        "Chickweed",
        "Crabgrass",
        "Dandelions",
      ];
      plantFilterStatus[plantNames[i - 1]] = checkbox.checked;
    }
  }

  console.log("Plant filtering status:", plantFilterStatus);
}

/**
 * Generate frames from video
 */
export async function generateFrames() {
  try {
    const input = "g";
    const response = await fetch("http://localhost:5000/run-test5", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ input: input }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const resultElement = document.getElementById("result");
    if (resultElement) {
      resultElement.innerText = "DONE";
    }
  } catch (error) {
    console.error("Error:", error);
    const resultElement = document.getElementById("result");
    if (resultElement) {
      resultElement.innerText = "Error occurred. Check console.";
    }
  }
}

/**
 * Generate images from specific frame
 */
export async function runTest() {
  try {
    const inputField = document.getElementById("inputField");
    if (!inputField) {
      console.error("Input field not found");
      return;
    }

    const input = inputField.value;
    const response = await fetch("http://localhost:5000/run-test5", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ input: input }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const resultElement = document.getElementById("result");
    if (resultElement) {
      resultElement.innerText = "GENERATED";
    }
  } catch (error) {
    console.error("Error:", error);
    const resultElement = document.getElementById("result");
    if (resultElement) {
      resultElement.innerText = "Error occurred. Check console.";
    }
  }
}

// Expose functions to window for use in HTML
window.generateFrames = generateFrames;
window.runTest = runTest;
