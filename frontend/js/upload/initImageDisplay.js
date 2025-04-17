export default function initImageDisplay() {
  // Initially hide all images
  const images = document.querySelectorAll("#panel-upload-data .image-container .image");
  images.forEach((img) => {
    img.style.display = "none";
  });

  // Add default event handling for checkboxes
  const checkboxes = document.querySelectorAll(
    '#panel-upload-data .checkbox-label input[type="checkbox"]'
  );
  checkboxes.forEach((checkbox) => {
    checkbox.checked = false;
  });

  // Set initial z-index values for proper layering
  const image1 = document.getElementById("image1");
  const image2 = document.getElementById("image2");
  const image3 = document.getElementById("image3");
  const image4 = document.getElementById("image4");

  if (image1) image1.style.zIndex = "1";
  if (image2) image2.style.zIndex = "2";
  if (image3) image3.style.zIndex = "3";
  if (image4) image4.style.zIndex = "4";

  // Set z-index for plant images
  for (let i = 1; i <= 13; i++) {
    const plantImage = document.getElementById(`plantImage${i}`);
    if (plantImage) {
      plantImage.style.zIndex = `${i + 3}`;
    }
  }
}

