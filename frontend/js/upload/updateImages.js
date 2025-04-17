const plantNames = [
  "Blue Violets",
  "Broadleaf Plantains",
  "Common Ivy",
  "Common Purslane",
  "Eastern Poison Ivy",
  "Fallen Leaves",
  "Japanese Honeysuckle",
  "Oxeye Daisy",
  "Roundleaf greenbrier",
  "Virginia Creeper",
  "Chickweed",
  "Crabgrass",
  "Dandelions",
];

export default function updateImages() {
  // Get main image checkboxes
  const image1Checkbox = document.getElementById("image1Checkbox");
  const image2Checkbox = document.getElementById("image2Checkbox");
  const image3Checkbox = document.getElementById("image3Checkbox");
  const image4Checkbox = document.getElementById("image4Checkbox");
  // Get main images
  const image1 = document.getElementById("image1");
  const image2 = document.getElementById("image2");
  const image3 = document.getElementById("image3");
  const image4 = document.getElementById("image4");

  // Reset display for main images
  if (image1) image1.style.display = "none";
  if (image2) image2.style.display = "none";
  if (image3) image3.style.display = "none";
  if (image4) image4.style.display = "none";

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

  if (image4Checkbox && image4Checkbox.checked && image3) {
    image4.style.display = "block";
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
    if (image4 && image4.style.display === "block") {
      image4.style.opacity = opacity;
    }
  } else {
    if (image1) image1.style.opacity = "1";
    if (image2) image2.style.opacity = "1";
    if (image3) image3.style.opacity = "1";
    if (image4) image4.style.opacity = "1";
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
      plantFilterStatus[plantNames[i - 1]] = checkbox.checked;
    }
  }

  console.log("Plant filtering status:", plantFilterStatus);
}

