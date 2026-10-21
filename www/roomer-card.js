/**
 * Roomer Presence Card for Lovelace
 *
 * Shows which room each person (or pet) is in, how sure Roomer is,
 * and the top signals behind the guess.
 *
 * Installation:
 * 1. Copy this file to /config/www/roomer-card.js
 * 2. Add to Lovelace resources:
 *    url: /local/roomer-card.js
 *    type: module
 *
 * Usage:
 * type: custom:roomer-card
 * title: Who's where          # optional
 * theme: auto                 # optional: auto | light | dark
 * entities:
 *   - sensor.ben_room
 *   - entity: sensor.alex_room
 *     person: person.alex     # optional, for the avatar picture
 *     name: Alex              # optional display name
 * show_confidence: true
 * show_indicators: true
 */

const ROOMER_CARD_VERSION = '2.0.0';
const DEFAULT_TITLE = "Who's where";
const AWAY_STATES = ['away', 'not_home', 'unknown', 'unavailable', ''];
const FONT_LINK_ID = 'roomer-card-fonts';
const FONT_HREF =
  'https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600' +
  '&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,700&display=swap';

function ensureFonts() {
  try {
    if (document.getElementById(FONT_LINK_ID)) return;
    if (document.querySelector(`link[href="${FONT_HREF}"]`)) return;
    const link = document.createElement('link');
    link.id = FONT_LINK_ID;
    link.rel = 'stylesheet';
    link.href = FONT_HREF;
    document.head.appendChild(link);
  } catch (e) {
    /* fonts are a nicety; fall back silently */
  }
}

function esc(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/** Normalise an `entities` entry (string or object) to {entity, person, name}. */
function normaliseEntry(entry) {
  if (typeof entry === 'string') return { entity: entry };
  if (entry && typeof entry === 'object' && typeof entry.entity === 'string') return { ...entry };
  return null;
}

function slugFor(entityId) {
  const objectId = String(entityId).split('.').slice(1).join('.') || String(entityId);
  return objectId.replace(/_room$/, '');
}

function prettyRoom(room) {
  const s = String(room).replace(/_/g, ' ').trim();
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : 'Unknown';
}

function initials(name) {
  const parts = String(name).trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return '?';
  const letters = parts.length === 1 ? parts[0].slice(0, 1) : parts[0][0] + parts[1][0];
  return letters.toUpperCase();
}

function ago(ms) {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 48) return `${h}h ago`;
  return `${Math.round(h / 24)}d ago`;
}

const STYLES = `
  :host { display: block; }
  ha-card.rc {
    --rc-bg: #FBF6EC; --rc-surface: #FFFFFF; --rc-ink: #2A2419; --rc-body: #4A4033;
    --rc-muted: #6B6150; --rc-line: #E7DCC5; --rc-line-strong: #D9CCB1;
    --rc-primary: #3E5C82; --rc-primary-ink: #31496A; --rc-tint1: #DCE4EE;
    --rc-secondary-ink: #47573A; --rc-tint2: #E4EADB;
    --rc-highlight-ink: #6F5213; --rc-tint3: #F7E4B5;
    --rc-font-body: 'DM Sans', var(--paper-font-body1_-_font-family, system-ui), sans-serif;
    --rc-font-display: 'Bricolage Grotesque', var(--paper-font-body1_-_font-family, system-ui), sans-serif;
    --ha-card-box-shadow: none;
    --ha-card-background: var(--rc-bg);
    --ha-card-border-radius: 14px;
    --ha-card-border-width: 1px;
    --ha-card-border-color: var(--rc-line);
    display: block;
    box-sizing: border-box;
    background: var(--rc-bg);
    border: 1px solid var(--rc-line);
    border-radius: 14px;
    box-shadow: none;
    padding: 20px;
    color: var(--rc-body);
    font-family: var(--rc-font-body);
    font-size: 14px;
    line-height: 1.4;
  }
  ha-card.rc.dark {
    --rc-bg: #1A1917; --rc-surface: #23211E; --rc-ink: #F2EEE8; --rc-body: #D3CEC6;
    --rc-muted: #A39C91; --rc-line: #34312C; --rc-line-strong: #46423B;
    --rc-primary: #93AECF; --rc-primary-ink: #AFC4DD; --rc-tint1: #21262E;
    --rc-secondary-ink: #B3C4A1; --rc-tint2: #252A22;
    --rc-highlight-ink: #E2C27F; --rc-tint3: #2D2A21;
  }
  .head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 14px; }
  .title {
    margin: 0; font-family: var(--rc-font-display); font-weight: 600; font-size: 22px;
    line-height: 1.2; color: var(--rc-ink); letter-spacing: -0.01em;
  }
  .count {
    flex: none; font-size: 12px; font-weight: 700; color: var(--rc-primary-ink);
    background: var(--rc-tint1); border-radius: 999px; padding: 3px 10px;
  }
  .rows { display: flex; flex-direction: column; gap: 10px; }
  .row {
    display: flex; align-items: center; gap: 12px;
    background: var(--rc-surface); border: 1px solid var(--rc-line);
    border-radius: 12px; padding: 12px 14px; min-width: 0;
  }
  .row.away { background: transparent; border: 1px dashed var(--rc-line-strong); }
  .avatar {
    flex: none; width: 44px; height: 44px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 15px; box-sizing: border-box; overflow: hidden;
  }
  .avatar img { width: 100%; height: 100%; object-fit: cover; border-radius: 50%; display: block; }
  .avatar.pic { border: 3px solid var(--rc-primary); background: var(--rc-surface); }
  .avatar.t1 { background: var(--rc-tint1); color: var(--rc-primary-ink); }
  .avatar.t2 { background: var(--rc-tint2); color: var(--rc-secondary-ink); }
  .avatar.t3 { background: var(--rc-tint3); color: var(--rc-highlight-ink); }
  .info { flex: 1; min-width: 0; }
  .name { font-size: 14px; font-weight: 700; color: var(--rc-body); }
  .room {
    font-family: var(--rc-font-display); font-weight: 600; font-size: 20px; line-height: 1.2;
    color: var(--rc-primary-ink); letter-spacing: -0.01em;
  }
  .row.away .room { color: var(--rc-muted); }
  .why {
    font-size: 12px; color: var(--rc-muted); margin-top: 2px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .chip {
    flex: none; font-size: 12px; font-weight: 700; border-radius: 6px; padding: 4px 8px;
    white-space: nowrap;
  }
  .chip.sure { color: var(--rc-secondary-ink); background: var(--rc-tint2); }
  .chip.pretty { color: var(--rc-highlight-ink); background: var(--rc-tint3); }
  .chip.guess { color: var(--rc-muted); background: var(--rc-line); }
  .missing { font-size: 13px; color: var(--rc-muted); }
  .foot { margin-top: 14px; font-size: 12px; color: var(--rc-muted); }
`;

class RoomerCard extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._sig = null;
    this._lastUpdated = null;
  }

  set hass(hass) {
    this._hass = hass;
    this._render();
  }

  get hass() {
    return this._hass;
  }

  setConfig(config) {
    if (!config || !Array.isArray(config.entities) || !config.entities.length) {
      throw new Error('Please define entities');
    }
    const entries = config.entities.map(normaliseEntry);
    if (entries.some((e) => !e)) {
      throw new Error('Each entry in entities must be an entity ID or an object with an `entity` key');
    }
    const theme = config.theme || 'auto';
    if (!['auto', 'light', 'dark'].includes(theme)) {
      throw new Error('theme must be one of: auto, light, dark');
    }
    this._config = config;
    this._entries = entries;
    this._theme = theme;
    this._sig = null;
    ensureFonts();
    this._render();
  }

  connectedCallback() {
    this._timer = setInterval(() => this._updateFooter(), 15000);
  }

  disconnectedCallback() {
    clearInterval(this._timer);
  }

  _isDark() {
    if (this._theme === 'dark') return true;
    if (this._theme === 'light') return false;
    const themes = this._hass && this._hass.themes;
    if (themes && typeof themes.darkMode === 'boolean') return themes.darkMode;
    return !!(window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
  }

  _updateFooter() {
    const foot = this.shadowRoot && this.shadowRoot.querySelector('.foot');
    if (foot && this._lastUpdated) foot.textContent = `Updated ${ago(Date.now() - this._lastUpdated)}`;
  }

  _render() {
    if (!this._hass || !this._config) return;
    const states = this._hass.states || {};
    const cfg = this._config;
    const showConfidence = cfg.show_confidence !== false;
    const showIndicators = cfg.show_indicators !== false;
    const dark = this._isDark();

    // Skip re-rendering when nothing this card shows has changed.
    const sig = JSON.stringify([
      dark,
      this._entries.map((e) => {
        const s = states[e.entity];
        const personId = e.person || `person.${slugFor(e.entity)}`;
        const p = states[personId];
        return [
          s ? s.state : null,
          s ? s.last_updated : null,
          s ? s.attributes : null,
          p ? p.attributes.entity_picture : null,
          p ? p.attributes.friendly_name : null,
        ];
      }),
    ]);
    if (sig === this._sig) {
      this._updateFooter();
      return;
    }
    this._sig = sig;

    let homeCount = 0;
    let latest = null;
    const rows = this._entries.map((entry, index) => {
      const state = states[entry.entity];
      if (!state) {
        return `<div class="row away"><div class="missing">Entity not found: ${esc(entry.entity)}</div></div>`;
      }
      const attrs = state.attributes || {};
      const personId = entry.person || `person.${slugFor(entry.entity)}`;
      const person = states[personId];
      const pAttrs = (person && person.attributes) || {};

      const fallbackName = String(attrs.friendly_name || slugFor(entry.entity).replace(/_/g, ' '))
        .replace(/\s+room$/i, '');
      const name = entry.name || pAttrs.friendly_name || fallbackName;

      const raw = String(state.state ?? '').toLowerCase();
      const away = AWAY_STATES.includes(raw);
      if (!away) homeCount += 1;
      const roomLabel = raw === 'away' || raw === 'not_home' ? 'Away'
        : raw === 'unknown' || raw === 'unavailable' || raw === '' ? 'Unknown'
          : prettyRoom(state.state);

      const ts = Date.parse(state.last_updated || state.last_changed || '');
      if (!Number.isNaN(ts) && (latest === null || ts > latest)) latest = ts;

      const pic = pAttrs.entity_picture;
      const avatar = pic
        ? `<div class="avatar pic"><img src="${esc(pic)}" alt=""></div>`
        : `<div class="avatar t${(index % 3) + 1}" aria-hidden="true">${esc(initials(name))}</div>`;

      let why = '';
      const indicators = Array.isArray(attrs.indicators) ? attrs.indicators : [];
      if (showIndicators && indicators.length) {
        const text = indicators.slice(0, 2).map((i) => String(i)).join(' · ');
        why = `<div class="why" title="${esc(text)}">${esc(text)}</div>`;
      }

      let chip = '';
      if (showConfidence && !away) {
        const conf = Math.min(1, Math.max(0, Number(attrs.confidence) || 0));
        const pct = Math.round(conf * 100);
        const [cls, word] = conf >= 0.8 ? ['sure', 'Sure'] : conf >= 0.5 ? ['pretty', 'Pretty sure'] : ['guess', 'Guessing'];
        chip = `<span class="chip ${cls}">${word} · ${pct}%</span>`;
      }

      return `
        <div class="row${away ? ' away' : ''}">
          ${avatar}
          <div class="info">
            <div class="name">${esc(name)}</div>
            <div class="room">${esc(roomLabel)}</div>
            ${why}
          </div>
          ${chip}
        </div>`;
    });

    this._lastUpdated = latest;
    const title = cfg.title || DEFAULT_TITLE;
    const foot = latest !== null ? `<div class="foot">Updated ${ago(Date.now() - latest)}</div>` : '';

    this.shadowRoot.innerHTML = `
      <style>${STYLES}</style>
      <ha-card class="rc${dark ? ' dark' : ''}">
        <div class="head">
          <h2 class="title">${esc(title)}</h2>
          <span class="count">${homeCount} home</span>
        </div>
        <div class="rows">${rows.join('')}</div>
        ${foot}
      </ha-card>`;
  }

  static getConfigElement() {
    return document.createElement('roomer-card-editor');
  }

  static getStubConfig(hass) {
    let entities = ['sensor.example_room'];
    if (hass && hass.states) {
      const found = Object.keys(hass.states).filter((id) => /^sensor\..+_room$/.test(id)).slice(0, 3);
      if (found.length) entities = found;
    }
    return {
      title: DEFAULT_TITLE,
      entities,
      show_confidence: true,
      show_indicators: true,
      theme: 'auto',
    };
  }

  getCardSize() {
    return (this._config?.entities?.length || 1) + 1;
  }
}

// Card editor for the GUI. Uses Home Assistant's <ha-form> with plain DOM
// properties and event listeners (no lit template bindings).
const EDITOR_SCHEMA = [
  { name: 'title', selector: { text: {} } },
  { name: 'entities', selector: { entity: { multiple: true, filter: { domain: 'sensor' } } } },
  {
    name: 'theme',
    selector: {
      select: {
        mode: 'dropdown',
        options: [
          { value: 'auto', label: 'Auto (follow Home Assistant)' },
          { value: 'light', label: 'Light' },
          { value: 'dark', label: 'Dark' },
        ],
      },
    },
  },
  { name: 'show_confidence', selector: { boolean: {} } },
  { name: 'show_indicators', selector: { boolean: {} } },
];

const EDITOR_LABELS = {
  title: 'Title',
  entities: 'Room sensors',
  theme: 'Theme',
  show_confidence: 'Show confidence',
  show_indicators: 'Show indicators',
};

class RoomerCardEditor extends HTMLElement {
  setConfig(config) {
    this._config = { ...config };
    this._update();
  }

  set hass(hass) {
    this._hass = hass;
    this._update();
  }

  _formData() {
    const c = this._config || {};
    return {
      title: c.title || '',
      entities: (c.entities || []).map(normaliseEntry).filter(Boolean).map((e) => e.entity),
      theme: c.theme || 'auto',
      show_confidence: c.show_confidence !== false,
      show_indicators: c.show_indicators !== false,
    };
  }

  _update() {
    if (!this._config) return;
    if (!this._form) {
      this._form = document.createElement('ha-form');
      this._form.computeLabel = (s) => EDITOR_LABELS[s.name] || s.name;
      this._form.addEventListener('value-changed', (ev) => this._valueChanged(ev.detail.value));
      this.appendChild(this._form);
    }
    this._form.schema = EDITOR_SCHEMA;
    this._form.data = this._formData();
    if (this._hass) this._form.hass = this._hass;
  }

  _valueChanged(value) {
    if (!value) return;
    // Keep per-entity objects (person/name overrides) for entities still selected.
    const previous = new Map(
      (this._config.entities || []).map(normaliseEntry).filter(Boolean).map((e) => [e.entity, e]),
    );
    const entities = (value.entities || []).map((id) => {
      const prev = previous.get(id);
      return prev && Object.keys(prev).length > 1 ? prev : id;
    });
    const config = { ...this._config, ...value, entities };
    if (!config.title) delete config.title;
    if (config.theme === 'auto') delete config.theme;
    this._config = config;
    this.dispatchEvent(new CustomEvent('config-changed', {
      detail: { config },
      bubbles: true,
      composed: true,
    }));
  }
}

if (!customElements.get('roomer-card')) customElements.define('roomer-card', RoomerCard);
if (!customElements.get('roomer-card-editor')) customElements.define('roomer-card-editor', RoomerCardEditor);

// Register with Lovelace
window.customCards = window.customCards || [];
if (!window.customCards.some((c) => c.type === 'roomer-card')) {
  window.customCards.push({
    type: 'roomer-card',
    name: 'Roomer Presence Card',
    description: "Who's where: each person's room, how sure Roomer is, and why",
    preview: true,
  });
}

console.info(`%c Roomer Card %c v${ROOMER_CARD_VERSION} `,
  'background: #3E5C82; color: white; padding: 2px 6px; border-radius: 3px 0 0 3px;',
  'background: #4A4033; color: white; padding: 2px 6px; border-radius: 0 3px 3px 0;'
);
