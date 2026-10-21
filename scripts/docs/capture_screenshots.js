#!/usr/bin/env node
/**
 * Regenerate the screenshots in docs/assets/ from the REAL dashboard and card code, using demo data.
 *
 *   npm install                                  # once, from the repo root (uses @playwright/test)
 *   node scripts/docs/capture_screenshots.js     # writes to docs/assets/
 *   node scripts/docs/capture_screenshots.js OUT_DIR
 *
 * If Playwright's own Chromium is not installed, point at any Chromium/Chrome:
 *   CHROMIUM_PATH=/path/to/chrome node scripts/docs/capture_screenshots.js
 *
 * How it works: a tiny local server stands in for Home Assistant. It serves dashboard/ untouched and answers
 * GET /api/states with a made-up household (Alice, Bob and the dog Max). The Lovelace card is rendered in a
 * bare page with a stand-in <ha-card> element and Home Assistant-like theme colours. Nothing here talks to a real
 * Home Assistant, so the pictures are illustrations, not captures of a live home.
 */
const http = require('http');
const fs = require('fs');
const path = require('path');
const { chromium } = require('@playwright/test');

const REPO = path.resolve(__dirname, '..', '..');
const DASHBOARD = path.join(REPO, 'dashboard');
const CARD_JS = path.join(REPO, 'www', 'whollm-card.js');
const OUT = path.resolve(process.argv[2] || path.join(REPO, 'docs', 'assets'));

// ---- demo data ---------------------------------------------------------------------------------------------
let scene = 0;

function sensor(id, friendly, room, confidence, indicators, raw, type, agoSeconds) {
  const t = new Date(Date.now() - agoSeconds * 1000).toISOString();
  return {
    entity_id: id, state: room, last_changed: t, last_updated: t,
    attributes: { friendly_name: friendly, confidence, indicators, raw_response: raw, entity_type: type },
  };
}

const SCENES = [
  // earlier: everyone somewhere else, so the timeline can show real "moved from -> to" events
  () => [
    sensor('sensor.alice_room', 'Alice Room', 'kitchen', 0.58, ['light: kitchen_light', 'motion: kitchen_motion', 'LLM: kitchen'], 'ROOM: kitchen', 'person', 900),
    sensor('sensor.bob_room', 'Bob Room', 'living_room', 0.71, ['media: living_room_tv', 'light: living_room_lamp', 'LLM: living_room'], 'ROOM: living_room', 'person', 1800),
    sensor('sensor.max_room', 'Max Room', 'bedroom', 0.42, ['LLM: bedroom'], 'ROOM: bedroom', 'pet', 3600),
  ],
  // later
  () => [
    sensor('sensor.alice_room', 'Alice Room', 'office', 0.82, ['motion: office_motion', 'computer: alice_pc', 'LLM: office'], 'ROOM: office', 'person', 40),
    sensor('sensor.bob_room', 'Bob Room', 'living_room', 0.71, ['media: living_room_tv', 'light: living_room_lamp', 'LLM: living_room'], 'ROOM: living_room', 'person', 2100),
    sensor('sensor.max_room', 'Max Room', 'kitchen', 0.46, ['motion: kitchen_motion', 'LLM: kitchen'], 'ROOM: kitchen', 'pet', 25),
  ],
];

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.png': 'image/png' };

function startMockHomeAssistant() {
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://x');
    if (url.pathname === '/api/states') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify(SCENES[scene]()));
    }
    if (url.pathname === '/__scene') {
      scene = Number(url.searchParams.get('n') || 0);
      res.writeHead(200);
      return res.end('ok');
    }
    const file = path.join(DASHBOARD, url.pathname === '/' ? 'index.html' : url.pathname);
    if (!file.startsWith(DASHBOARD) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
      res.writeHead(404);
      return res.end('not found');
    }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
    fs.createReadStream(file).pipe(res);
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(server)));
}

// ---- the Lovelace card harness -----------------------------------------------------------------------------
const CARD_PAGE = `<!doctype html><html><head><meta charset="utf-8"><style>
  :root{--primary-color:#03a9f4;--card-background-color:#1c1c1c;--primary-text-color:#e1e1e1;--secondary-text-color:#9b9b9b;
    --divider-color:rgba(225,225,225,.12);--secondary-background-color:#202020;--success-color:#43a047;--warning-color:#ffa600;
    --error-color:#db4437;--text-primary-color:#fff}
  body{margin:0;padding:20px;background:#111;font-family:Roboto,"Segoe UI",sans-serif;color:var(--primary-text-color)}
  ha-card{display:block;background:var(--card-background-color);border-radius:12px;box-shadow:0 2px 6px rgba(0,0,0,.5);overflow:hidden}
  .ha-card-header{font-size:24px;line-height:32px;padding:24px 16px 16px}
</style></head><body>
<script>
  // stand-in for Home Assistant's <ha-card>: draws the header the way HA does
  customElements.define('ha-card', class extends HTMLElement {
    connectedCallback() {
      const h = this.getAttribute('header');
      if (h && !this.querySelector('.ha-card-header')) {
        const d = document.createElement('div'); d.className = 'ha-card-header'; d.textContent = h; this.prepend(d);
      }
    }
  });
</script>
<script src="/__card.js"></script>
<script>
  const card = document.createElement('whollm-card');
  card.setConfig({ title: 'Who is where', entities: ['sensor.alice_room', 'sensor.bob_room', 'sensor.max_room', 'sensor.carol_room'] });
  document.body.appendChild(card);
  const mk = (name, room, conf, ind, type) => ({ state: room, attributes: { friendly_name: name, confidence: conf, indicators: ind, entity_type: type } });
  card.hass = { states: {
    'sensor.alice_room': mk('Alice Room', 'office', 0.82, ['motion: office_motion', 'computer: alice_pc', 'LLM: office'], 'person'),
    'sensor.bob_room': mk('Bob Room', 'living_room', 0.71, ['media: living_room_tv', 'light: living_room_lamp', 'LLM: living_room'], 'person'),
    'sensor.max_room': mk('Max Room', 'kitchen', 0.46, ['motion: kitchen_motion', 'LLM: kitchen'], 'pet'),
    'sensor.carol_room': mk('Carol Room', 'away', 0.95, ['device_tracker: carol_iphone'], 'person'),
  } };
</script></body></html>`;

// ---- main --------------------------------------------------------------------------------------------------
(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const server = await startMockHomeAssistant();
  const base = `http://127.0.0.1:${server.address().port}`;
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_PATH || undefined,
    args: ['--no-sandbox'],
  });

  try {
    // 1. sign-in screen (shows a neutral example address, not a real one)
    {
      const ctx = await browser.newContext({ viewport: { width: 1100, height: 700 }, deviceScaleFactor: 1 });
      const page = await ctx.newPage();
      await page.goto(base + '/');
      await page.fill('#ha-url', 'http://homeassistant.local:8123');
      await page.waitForTimeout(400);
      await page.screenshot({ path: path.join(OUT, 'dashboard-signin.png') });
      await ctx.close();
    }

    // 2. live dashboard
    {
      const ctx = await browser.newContext({ viewport: { width: 1100, height: 1095 }, deviceScaleFactor: 1 });
      await ctx.addInitScript(() => {
        localStorage.setItem('ha_manual_token', 'demo-token');
        localStorage.setItem('ha_url', 'http://homeassistant.local:8123');
      });
      const page = await ctx.newPage();
      await page.request.get(base + '/__scene?n=0');
      await page.goto(base + '/');
      await page.waitForSelector('.person-card', { timeout: 15000 });
      await page.request.get(base + '/__scene?n=1');
      await page.evaluate(() => fetchStates()); // the dashboard's own poll function
      await page.waitForTimeout(800);
      await page.screenshot({ path: path.join(OUT, 'dashboard-live.png') });
      await ctx.close();
    }

    // 3. Lovelace card
    {
      const ctx = await browser.newContext({ viewport: { width: 460, height: 640 }, deviceScaleFactor: 2 });
      const page = await ctx.newPage();
      await page.route(base + '/__card.js', r => r.fulfill({ contentType: 'text/javascript', body: fs.readFileSync(CARD_JS, 'utf8') }));
      await page.route(base + '/__card', r => r.fulfill({ contentType: 'text/html', body: CARD_PAGE }));
      await page.goto(base + '/__card');
      await page.waitForSelector('.entity');
      await page.waitForTimeout(500);
      const box = await page.locator('ha-card').boundingBox();
      await page.screenshot({
        path: path.join(OUT, 'lovelace-card.png'),
        clip: { x: box.x - 20, y: box.y - 20, width: box.width + 40, height: box.height + 40 },
      });
      await ctx.close();
    }
  } finally {
    await browser.close();
    server.close();
  }
  console.log('Wrote dashboard-signin.png, dashboard-live.png, lovelace-card.png to', OUT);
})().catch(err => { console.error(err); process.exit(1); });
