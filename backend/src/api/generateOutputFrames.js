import { spawn } from "node:child_process";

export default async function generateOutputFrames(input) {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn("python", ["./app.py", input]);

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
