export default async function uploadVideo(event) {
  event.preventDefault();
  console.log("Upload form submitted");

  const fileInput = event.target.querySelector('input[type="file"]');

  if (fileInput.files.length === 0) {
    console.log("Please select a video file first.");
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
      throw new Error("Upload failed");
    }

    console.log("Upload successful:", uploadResult);

    // Process the uploaded file
    const processResponse = await fetch("http://localhost:5000/api/generateOutputFrames", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: filePath }),
    });

    const processResult = await processResponse.json();

    if (!processResult.success) {
      throw new Error("Processing failed");
    }

    console.log("Processing result:", processResult);
  } catch (error) {
    console.error("Upload error:", error);
  }
}
