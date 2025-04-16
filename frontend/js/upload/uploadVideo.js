export default async function uploadVideo(event) {
  const messagebox = document.querySelector(".show-message");

  // -------------- uploadVideo --------------
  event.preventDefault();

  const fileInput = event.target.querySelector('input[type="file"]');

  if (fileInput.files.length === 0) {
    messagebox.textContent = "Please select a video file first.";
    return;
  }

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("file", file);

  messagebox.textContent = "Starting upload video: " + file.name;

  try {
    // Upload the file to the backend
    const uploadResponse = await fetch("http://localhost:5000/api/upload", {
      method: "POST",
      body: formData,
    });

    const uploadResult = await uploadResponse.json();

    if (!uploadResult.success) {
      console.error("Upload failed: ", uploadResult.message);
      messagebox.textContent = "Upload failed. Please try again.";
      return;
    }

    console.log("Upload successful:", uploadResult.path);
    messagebox.textContent = "Upload successful, video processing started.";

    // -------------- processVideo --------------
    const videoFilename = uploadResult.path.split(/[\\/]/).pop();

    console.log("Video filename:", videoFilename);

    const processResponse = await fetch("http://localhost:5000/api/generateOutputFrames", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ videoFilename }),
    });

    const processResult = await processResponse.json();

    if (!processResult.success) {
      messagebox.textContent = "Processing failed. Please try again.";
      return;
    }
    messagebox.textContent = "Processing successful!";
    console.log(`\x1b[36m%s\x1b[0m`, "Processing result:");
    console.log(`\x1b[36m%s\x1b[0m`, processResult.message);
  } catch (error) {
    messagebox.textContent = "An error occurred during the upload or processing. Please try again.";
    console.error("Upload error:", error);
  }
}
