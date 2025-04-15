import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default async function generateOutputFrames(videoFilename, mode = "frames") {
  return new Promise((resolve, reject) => {
    const scriptPath = path.resolve(__dirname, "../../../python/app.py");
    const pythonProcess = spawn("python", [scriptPath, mode, videoFilename]);

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
