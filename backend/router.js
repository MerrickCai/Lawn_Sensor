import { parseBody, send } from "./util.js";

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
    // Home route
    "GET /": async () => {
      send(res, 200, {
        success: true,
        message: `Welcome back!`,
      });
    },
    // General route
    "POST /api/": async () => {
      const { body } = await parseBody(req);
      const result = await get(body);
      send(res, result.error ? 400 : 200, result);
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
