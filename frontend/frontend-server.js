import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { WebSocketServer, WebSocket } from "ws";
import chokidar from "chokidar";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const HOST = "localhost";
const PORT = 3000;

const mimeTypes = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".css": "text/css",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".json": "application/json",
};

// ------------------------- HTTP server -------------------------
const server = http.createServer((req, res) => {
  const rawPath = req.url === "/" ? "/index.html" : req.url;
  const urlPath = decodeURIComponent(rawPath);
  const filePath = path.normalize(path.join(__dirname, urlPath));

  // Prevent directory traversal attacks
  // Ensure the requested file is within the server directory
  if (!filePath.startsWith(__dirname)) {
    res.writeHead(403, { "Content-Type": "text/plain" });
    res.end("403 Forbidden");
    return;
  }

  const ext = path.extname(filePath);

  fs.readFile(filePath, (err, content) => {
    if (err) {
      res.writeHead(404, { "Content-Type": "text/plain" });
      res.end("404 Not Found");
    } else {
      // Inject WebSocket client script for hot reload
      if (ext === ".html") {
        const injectedContent = content.toString().replace(
          "</body>",
          `<script>
            const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host);
            ws.onmessage = (message) => {
              if (message.data === "reload") {
                location.reload();
              }
            };
          </script></body>`
        );
        res.writeHead(200, { "Content-Type": mimeTypes[ext] });
        res.end(injectedContent);
      } else {
        res.writeHead(200, { "Content-Type": mimeTypes[ext] || "application/octet-stream" });
        res.end(content);
      }
    }
  });
});

// ------------------------- WebSocket server -------------------------
const wss = new WebSocketServer({ server });

wss.on("connection", () => {
  console.log("Hot‑reload client connected\n");
});

// ------------------------- Watch file changes -------------------------
chokidar
  .watch(__dirname, { ignoreInitial: true, ignored: /node_modules/ })
  .on("change", (filePath) => {
    console.log(`\x1b[36m%s\x1b[0m`, `File changed: ${filePath}`);
    wss.clients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        client.send("reload");
      }
    });
  });

// ------------------------- Start the server -------------------------
server.listen(PORT, HOST, () => {
  console.log(`Frontend server running at http://${HOST}:${PORT}`);
});
