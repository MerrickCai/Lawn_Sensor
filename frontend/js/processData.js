// import leaflet library
import "../Leaflet/leaflet.js";
import buildCharts from "./component/chart.js";

export default function processData(data) {
  const tableBody = document.querySelector("#data-table tbody");
  const gallery = document.querySelector("#gallery");
  const rowsPerPage = 5; // Number of rows per page
  let currentPage = 1; // Current page

  // Function to render table rows for the current page
  function renderTable(page, sortedData = data) {
    tableBody.innerHTML = ""; // Clear existing rows
    const start = (page - 1) * rowsPerPage;
    const end = start + rowsPerPage;
    const pageData = sortedData.slice(start, end);

    pageData.forEach((item) => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td><img src="${item.image_path}" class="table-img" alt="${item.image_name}"></td>
        <td>${item.image_name}</td>
        <td>${item.predicted_class}</td>
        <td>${item.confidence}</td>
        <td>${item.gps.lat || "N/A"}</td>
        <td>${item.gps.long || "N/A"}</td>
      `;
      tableBody.appendChild(row);
    });
  }

  // Function to update pagination controls
  function updatePagination() {
    const totalPages = Math.ceil(data.length / rowsPerPage);
    const paginationContainer = document.querySelector("#pagination");
    paginationContainer.innerHTML = ""; // Clear existing buttons

    // Previous button
    const prevButton = document.createElement("button");
    prevButton.textContent = "Previous";
    prevButton.disabled = currentPage === 1;
    prevButton.addEventListener("click", () => {
      if (currentPage > 1) {
        currentPage--;
        renderTable(currentPage);
        updatePagination();
      }
    });
    paginationContainer.appendChild(prevButton);

    // Page numbers
    for (let i = 1; i <= totalPages; i++) {
      const pageButton = document.createElement("button");
      pageButton.textContent = i;
      pageButton.className = i === currentPage ? "active" : "";
      pageButton.addEventListener("click", () => {
        currentPage = i;
        renderTable(currentPage);
        updatePagination();
      });
      paginationContainer.appendChild(pageButton);
    }

    // Next button
    const nextButton = document.createElement("button");
    nextButton.textContent = "Next";
    nextButton.disabled = currentPage === totalPages;
    nextButton.addEventListener("click", () => {
      if (currentPage < totalPages) {
        currentPage++;
        renderTable(currentPage);
        updatePagination();
      }
    });
    paginationContainer.appendChild(nextButton);
  }

  // Initial render
  renderTable(currentPage);
  updatePagination();

  let sumLat = 0,
    sumLng = 0;
  const classCounts = {};
  const markers = []; // Array to store marker references

  data.forEach((item, index) => {
    // Accumulate coordinates for average
    sumLat += item.gps.lat;
    sumLng += item.gps.long;

    // Tally counts for chart (plant species distribution)
    classCounts[item.predicted_class] = (classCounts[item.predicted_class] || 0) + 1;

    // Create gallery item with image and button
    const imageContainer = document.createElement("div");
    imageContainer.classList.add("image-container");
    imageContainer.dataset.index = index; // Store the index for reference
    imageContainer.innerHTML = `
            <img src="${item.image_path}" alt="${item.predicted_class}">
            <p><strong>${item.predicted_class}</strong></p>
            <button class="show-on-map-btn">Show on Map</button>
          `;
    gallery.appendChild(imageContainer);
  });

  // --- Initialize Leaflet Map ---
  const avgLat = sumLat / data.length;
  const avgLng = sumLng / data.length;
  const map = L.map("map").setView([avgLat, avgLng], 15);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);

  // Add markers to the map
  data.forEach((item, index) => {
    // Create a custom icon using the image
    const customIcon = L.divIcon({
      className: "custom-marker", // Custom class for styling
      html: `<img src="${item.image_path}" alt="${item.predicted_class}" class="marker-img">`,
      iconSize: [50, 50], // Adjust the size of the icon
      iconAnchor: [25, 25], // Anchor the icon at its center
    });

    // Create a marker with the custom icon
    const marker = L.marker([item.gps.lat, item.gps.long], { icon: customIcon })
      .addTo(map)
      .bindPopup(
        `
        <div class="custom-popup">
          <h3>${item.predicted_class}</h3>
          <div class="popup-content">
            <div class="popup-info">
            </div>
            <div class="popup-image-container">
              <img src="${item.image_path}" alt="${item.predicted_class}">
            </div>
          </div>
        </div>
        `,
        {
          maxWidth: 300,
          className: "custom-popup-container",
        }
      );

    // Store marker reference in array
    markers.push(marker);
  });

  // Add click event listeners to the Show on Map buttons in gallery
  document.querySelectorAll(".show-on-map-btn").forEach((button) => {
    button.addEventListener("click", function () {
      const containerElement = this.closest(".image-container");
      const index = parseInt(containerElement.dataset.index);

      if (markers[index]) {
        // Open the popup for the corresponding marker
        map.setView([data[index].gps.lat, data[index].gps.long], 18);
        markers[index].openPopup();
      }
    });
  });

  const mapStyleSelector = document.getElementById("map-style");
  mapStyleSelector.addEventListener("change", function (event) {
    const newTileLayer = event.target.value;

    // Remove the current tile layer
    map.eachLayer((layer) => {
      if (layer instanceof L.TileLayer) {
        map.removeLayer(layer);
      }
    });

    // Add the new tile layer
    L.tileLayer(newTileLayer, {
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(map);
  });

  buildCharts(classCounts);

  const modal = document.getElementById("image-modal");
  const modalImage = document.getElementById("modal-image");
  const closeModal = document.querySelector(".close");

  data.forEach((item) => {
    const customIcon = L.divIcon({
      className: "custom-marker",
      html: `<img src="${item.image_path}" alt="${item.predicted_class}" class="marker-img">`,
      iconSize: [50, 50],
      iconAnchor: [25, 25],
    });

    const marker = L.marker([item.gps.lat, item.gps.long], {
      icon: customIcon,
    }).addTo(map);

    // Add click event to open modal
    marker.on("click", () => {
      modal.style.display = "block";
      modalImage.src = item.image_path;
    });
  });

  // Close modal
  closeModal.addEventListener("click", () => {
    modal.style.display = "none";
  });

  const totalPlantsElement = document.getElementById("total-plants");
  const averageHealthElement = document.getElementById("average-health");
  const uniqueSpeciesElement = document.getElementById("unique-species");


  const totalPlants = data.length;
  const averageHealth = (
    data.reduce((sum, item) => sum + item.plantHealth, 0) / totalPlants
  ).toFixed(2);
  const averageTGI = (
    data.reduce((sum, item) => sum + item.average_TGI, 0) / 1
  );
  const uniqueSpecies = new Set(data.map((item) => item.predicted_class)).size;

  totalPlantsElement.textContent = `Total Plants: ${totalPlants}`;
  averageHealthElement.textContent = `Average Triangular Greenness Index: ${(averageTGI).toFixed(2)}`;
  uniqueSpeciesElement.textContent = `Unique Species: ${uniqueSpecies}`;
}

document.querySelector("#table-search").addEventListener("input", function (event) {
  const searchTerm = event.target.value.toLowerCase();
  const rows = document.querySelectorAll("#data-table tbody tr");

  rows.forEach((row) => {
    const imageName = row.children[1].textContent.toLowerCase();
    const plantSpecies = row.children[2].textContent.toLowerCase();

    if (imageName.includes(searchTerm) || plantSpecies.includes(searchTerm)) {
      row.style.display = ""; // Show row
    } else {
      row.style.display = "none"; // Hide row
    }
  });
});
