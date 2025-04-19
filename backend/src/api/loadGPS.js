import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "../../..");
const scriptPath = path.join(projectRoot, "python", "app.py");

export default async function loadGPS(frame_index, mode = "GPS") {
  return new Promise((resolve, reject) => {
    console.log(
      `\x1b[36m%s\x1b[0m`,
      `Running from: ${projectRoot} | Calling Python script: python ${scriptPath} ${mode} ${frame_index}`
    );

    const pythonProcess = spawn("python", [scriptPath, mode, frame_index], {
      cwd: projectRoot,
    });

    console.log("Python process started with PID:", pythonProcess.pid);

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
