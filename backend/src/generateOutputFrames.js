import { spawn } from "child_process";

export async function generateOutputFrames(body) {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn("python", ["./app.py", body]);

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
