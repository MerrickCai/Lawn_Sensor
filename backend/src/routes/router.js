import { parseBody, send, parseUpload } from "../utils/utils.js";
import generateOutputFrames from "../api/generateOutputFrames.js";
import generateImages from "../api/generateImages.js";

export async function handleRequest(req, res) {
  // -------------------------- Initialization --------------------------

  res.setHeader("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  res.setHeader("Access-Control-Max-Age", "86400");

  const origin = req.headers.origin;
  const allowedOrigin = origin === "http://localhost:3000" ? origin : null;

  // Set CORS headers for all responses
  if (allowedOrigin) {
    res.setHeader("Access-Control-Allow-Origin", allowedOrigin);
  }

  // Handle preflight request for CORS
  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  // -------------------------- Public Routes --------------------------

  const routes = {
    "GET /": async () => {
      send(res, 200, {
        success: true,
        message: `Welcome back!`,
      });
    },
    "POST /api/upload": async () => {
      const fileInfo = await parseUpload(req);
      send(res, 200, { success: true, path: fileInfo.filePath });
    },
    "POST /api/generateOutputFrames": async () => {
      console.log("Processing video...");
      const { videoFilename } = await parseBody(req);
      console.log("Video filename:", videoFilename);
      const result = await generateOutputFrames(videoFilename);
      send(res, result.success ? 200 : 400, result);
    },
    "POST /api/generateImages": async () => {
      console.log("Starting image generation...");
      const { frame_index } = await parseBody(req);
      console.log(`Generating images for frame: ${frame_index}`);
      const result = await generateImages(frame_index);
      console.log(`Image generation ${result.success ? "completed successfully" : "failed"}`);
      send(res, result.success ? 200 : 400, result);
    },
  };

  const routeKey = `${req.method} ${req.url}`;
  if (routes[routeKey]) {
    await routes[routeKey]();
    return;
  }

  // -------------------------- Fallback: 404 Not Found --------------------------

  send(res, 404, { success: false, message: "Route not found" });
}
