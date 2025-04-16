// ----------- Generate images from specific frame -----------
async function GenerateImages() {
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

    const resultElement = document.getElementById("show-result");
    if (resultElement) {
      resultElement.innerText = "GENERATED";
    }
  } catch (error) {
    console.error("Error:", error);
    const resultElement = document.getElementById("show-result");
    if (resultElement) {
      resultElement.innerText = "Error occurred. Check console.";
    }
  }
}

// ----------- Initialize image display functionality -----------
export default function initImageDisplay() {
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
  const generateImagesBtn = document.getElementById("generateImagesBtn");

  if (generateImagesBtn) {
    generateImagesBtn.addEventListener("click", GenerateImages);
  }
}
