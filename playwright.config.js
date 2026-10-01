import { defineConfig } from "@playwright/test";

const python = process.env.DUSTTWIN_TEST_PYTHON ?? ".venv/bin/python";
// The executable may be an absolute path containing spaces.
const quotedPython = `'${python.replace(/'/g, "'\\''")}'`;

export default defineConfig({
  testDir: "./tests/browser",
  fullyParallel: false,
  workers: 1,
  reporter: "list",
  use: { baseURL: "http://127.0.0.1:15174", browserName: "chromium" },
  webServer: [
    {
      command: `${quotedPython} scripts/serve.py --port 18100`,
      url: "http://127.0.0.1:18100/health",
      env: { DUSTTWIN_ALLOWED_ORIGINS: "http://127.0.0.1:15174" },
      timeout: 20000,
    },
    {
      command: `${quotedPython} scripts/serve_example.py --port 15174`,
      url: "http://127.0.0.1:15174/examples/",
      timeout: 20000,
    },
  ],
});
