import { spawn } from "node:child_process";

export default async function generateOutputFrames(videoFilename, mode = "frames") {
  return new Promise((resolve, reject) => {
    const scriptPath = "python/app.py";
    console.log("Calling Python script:", `python ${scriptPath} ${mode} ${videoFilename}`);
    const pythonProcess = spawn("python", [scriptPath, mode, videoFilename], {
      cwd: "../",
    });

    let outputData = "";

    pythonProcess.stdout.on("data", (data) => {
      outputData += data.toString();
    });

    pythonProcess.stderr.on("data", (data) => {
      console.error(`Python stderr: ${data}`);
    });

    pythonProcess.on("close", (code) => {
      if (code === 0) {
        resolve({ success: true, message: outputData.trim() });
      } else {
        reject({ success: false, message: "Python script failed" });
      }
    });
  });
}
