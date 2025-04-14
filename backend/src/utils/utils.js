import fs from "fs";
import path from "path";
import busboy from "busboy";

export function parseUpload(req) {
  return new Promise((resolve, reject) => {
    const bb = busboy({ headers: req.headers });
    const uploadDir = "../frontend/uploadVideo";

    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }

    let filePath = "";

    bb.on("file", (fieldname, file, info) => {
      const { filename } = info;
      const safeFilename = path.basename(filename);
      filePath = path.join(uploadDir, safeFilename);

      const saveTo = fs.createWriteStream(filePath);
      file.pipe(saveTo);

      saveTo.on("error", (err) => {
        console.error("File write error:", err);
        reject(err);
      });
    });

    bb.on("finish", () => {
      if (!filePath) {
        reject(new Error("No file uploaded"));
      } else {
        resolve({ filePath });
      }
    });

    bb.on("error", (err) => {
      console.error("Busboy error:", err);
      reject(err);
    });

    req.pipe(bb);
  });
}

export function parseBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch (err) {
        reject(err);
      }
    });
    req.on("error", reject);
  });
}

export function send(res, statusCode, payload, ContentType) {
  if (!ContentType) {
    ContentType = typeof payload === "object" ? "application/json" : "text/plain";
  }
  res.writeHead(statusCode, { "Content-Type": ContentType });
  res.end(typeof payload === "object" ? JSON.stringify(payload) : payload);
}
