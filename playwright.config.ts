import { defineConfig } from '@playwright/test';

// PW_CHANNEL=chrome makes Playwright drive an installed Google Chrome instead of
// its own Chromium build. CI uses it so the runner's preinstalled Chrome is
// enough and no browser download is needed.
const channel = process.env.PW_CHANNEL;

export default defineConfig({
  testDir: './e2e',
  timeout: 30000,
  retries: 0,
  forbidOnly: !!process.env.CI,
  reporter: process.env.CI ? [['github'], ['list']] : 'list',
  use: {
    baseURL: 'http://localhost:3380',
    headless: true,
    screenshot: 'only-on-failure',
    trace: 'on-first-retry',
  },
  projects: [
    { name: 'chromium', use: { browserName: 'chromium', ...(channel ? { channel } : {}) } },
  ],
  // The dashboard is a static page, so a plain file server is all it needs.
  webServer: {
    command: 'python3 -m http.server 3380 --directory dashboard',
    url: 'http://localhost:3380',
    reuseExistingServer: !process.env.CI,
  },
});
