import http from "node:http";
import { handleRequest } from "./src/routes/router.js";

const server = http.createServer(handleRequest);

server.on("error", (err) => {
  console.error("Server error:", err);
});

server.listen(5000, () => {
  console.log(`\x1b[36m%s\x1b[0m`, `Backend server running at http://localhost:5000/\n`);
});
