export default async function uploadVideo(event) {
  // -------------- uploadVideo --------------
  event.preventDefault();
  alert("Upload form submitted");

  const fileInput = event.target.querySelector('input[type="file"]');

  if (fileInput.files.length === 0) {
    alert("Please select a video file first.");
    return;
  }

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("file", file);

  console.log("Uploading file:", file);

  try {
    // Upload the file to the backend
    const uploadResponse = await fetch("http://localhost:5000/api/upload", {
      method: "POST",
      body: formData,
    });

    const uploadResult = await uploadResponse.json();

    if (!uploadResult.success) {
      console.error("Upload failed:");
      alert("Upload failed. Please try again.");
      return;
    }

    console.log("Upload successful:", uploadResult.path);
    alert("Upload successful!");

    // -------------- processVideo --------------
    const videoFilename = uploadResult.path.split(/[\\/]/).pop();

    const processResponse = await fetch("http://localhost:5000/api/generateOutputFrames", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ videoFilename }),
    });

    const processResult = await processResponse.json();

    if (!processResult.success) {
      alert("Processing failed. Please try again.");
      return;
    }

    console.log("Processing result:", processResult);
    alert("Processing successful!");
  } catch (error) {
    alert("An error occurred during the upload or processing. Please try again.");
    console.error("Upload error:", error);
  }
}
