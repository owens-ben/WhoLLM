/**
 * WhoLLM Presence Card for Lovelace
 * 
 * A custom card to display room presence for people and pets.
 * 
 * Installation:
 * 1. Copy this file to /config/www/whollm-card.js
 * 2. Add to Lovelace resources:
 *    url: /local/whollm-card.js
 *    type: module
 * 
 * Usage:
 * type: custom:whollm-card
 * title: Room Presence
 * entities:
 *   - sensor.ben_room
 *   - sensor.alex_room
 */

class WhoLLMCard extends HTMLElement {
  set hass(hass) {
    this._hass = hass;
    this._render();
  }

  setConfig(config) {
    if (!config.entities) {
      throw new Error('Please define entities');
    }
    this._config = config;
  }

  _render() {
    if (!this._hass || !this._config) return;

    const entities = this._config.entities || [];
    const title = this._config.title || 'Room Presence';
    const showConfidence = this._config.show_confidence !== false;
    const showIndicators = this._config.show_indicators !== false;

    // Build HTML
    let html = `
      <ha-card header="${title}">
        <div class="card-content">
    `;

    entities.forEach(entityId => {
      const state = this._hass.states[entityId];
      if (!state) {
        html += `<div class="entity unknown">Entity not found: ${entityId}</div>`;
        return;
      }

      const name = state.attributes.friendly_name || entityId.split('.')[1];
      const room = state.state || 'unknown';
      const confidence = state.attributes.confidence || 0;
      const indicators = state.attributes.indicators || [];
      const source = state.attributes.source || 'llm';

      // Determine status class
      let statusClass = 'home';
      if (room === 'away') statusClass = 'away';
      else if (room === 'unknown') statusClass = 'unknown';

      // Room emoji/icon
      const roomIcons = {
        'office': '💻',
        'living_room': '🛋️',
        'bedroom': '🛏️',
        'kitchen': '🍳',
        'bathroom': '🚿',
        'entry': '🚪',
        'garage': '🚗',
        'away': '🚶',
        'unknown': '❓',
      };
      const icon = roomIcons[room.toLowerCase()] || '📍';

      html += `
        <div class="entity ${statusClass}">
          <div class="entity-row">
            <span class="icon">${icon}</span>
            <span class="name">${name}</span>
            <span class="room">${room.replace('_', ' ')}</span>
          </div>
      `;

      if (showConfidence) {
        const confPercent = Math.round(confidence * 100);
        const confClass = confidence >= 0.7 ? 'high' : confidence >= 0.4 ? 'medium' : 'low';
        html += `
          <div class="confidence ${confClass}">
            <div class="bar" style="width: ${confPercent}%"></div>
            <span class="label">${confPercent}%</span>
          </div>
        `;
      }

      if (showIndicators && indicators.length > 0) {
        html += `
          <div class="indicators">
            ${indicators.slice(0, 3).map(i => `<span class="chip">${i}</span>`).join('')}
          </div>
        `;
      }

      html += '</div>';
    });

    html += `
        </div>
      </ha-card>
      <style>
        ha-card {
          padding: 0;
        }
        .card-content {
          padding: 16px;
        }
        .entity {
          padding: 12px;
          margin-bottom: 8px;
          border-radius: 8px;
          background: var(--secondary-background-color);
        }
        .entity.away {
          opacity: 0.6;
        }
        .entity.unknown {
          opacity: 0.4;
        }
        .entity-row {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .icon {
          font-size: 24px;
        }
        .name {
          flex: 1;
          font-weight: 500;
        }
        .room {
          font-weight: 600;
          text-transform: capitalize;
          color: var(--primary-color);
        }
        .entity.away .room {
          color: var(--error-color);
        }
        .confidence {
          margin-top: 8px;
          height: 4px;
          background: var(--divider-color);
          border-radius: 2px;
          position: relative;
          overflow: hidden;
        }
        .confidence .bar {
          height: 100%;
          border-radius: 2px;
          transition: width 0.3s ease;
        }
        .confidence.high .bar {
          background: var(--success-color, #4caf50);
        }
        .confidence.medium .bar {
          background: var(--warning-color, #ff9800);
        }
        .confidence.low .bar {
          background: var(--error-color, #f44336);
        }
        .confidence .label {
          position: absolute;
          right: 0;
          top: -18px;
          font-size: 11px;
          color: var(--secondary-text-color);
        }
        .indicators {
          display: flex;
          flex-wrap: wrap;
          gap: 4px;
          margin-top: 8px;
        }
        .chip {
          font-size: 10px;
          padding: 2px 6px;
          background: var(--primary-color);
          color: var(--text-primary-color);
          border-radius: 10px;
          opacity: 0.8;
        }
      </style>
    `;

    this.innerHTML = html;
  }

  static getConfigElement() {
    return document.createElement('whollm-card-editor');
  }

  static getStubConfig() {
    return {
      title: 'Room Presence',
      entities: ['sensor.example_room'],
      show_confidence: true,
      show_indicators: true,
    };
  }

  getCardSize() {
    return (this._config?.entities?.length || 1) + 1;
  }
}

// Card Editor for GUI configuration
class WhoLLMCardEditor extends HTMLElement {
  setConfig(config) {
    this._config = config;
    this._render();
  }

  _render() {
    this.innerHTML = `
      <div class="editor">
        <ha-textfield
          label="Title"
          .value="${this._config.title || ''}"
          @input="${e => this._valueChanged('title', e.target.value)}"
        ></ha-textfield>
        <ha-entity-picker
          .hass="${this._hass}"
          .value="${this._config.entities?.[0] || ''}"
          .includeDomains="${['sensor']}"
          @value-changed="${e => this._valueChanged('entities', [e.detail.value])}"
          label="Person Entity"
        ></ha-entity-picker>
        <ha-switch
          .checked="${this._config.show_confidence !== false}"
          @change="${e => this._valueChanged('show_confidence', e.target.checked)}"
        >
          Show Confidence
        </ha-switch>
        <ha-switch
          .checked="${this._config.show_indicators !== false}"
          @change="${e => this._valueChanged('show_indicators', e.target.checked)}"
        >
          Show Indicators
        </ha-switch>
      </div>
      <style>
        .editor {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }
      </style>
    `;
  }

  _valueChanged(key, value) {
    this._config = { ...this._config, [key]: value };
    this.dispatchEvent(new CustomEvent('config-changed', { detail: { config: this._config } }));
  }

  set hass(hass) {
    this._hass = hass;
  }
}

customElements.define('whollm-card', WhoLLMCard);
customElements.define('whollm-card-editor', WhoLLMCardEditor);

// Register with Lovelace
window.customCards = window.customCards || [];
window.customCards.push({
  type: 'whollm-card',
  name: 'WhoLLM Presence Card',
  description: 'Display room presence for people and pets',
  preview: true,
});

console.info('%c WhoLLM Card %c v1.0.0 ', 
  'background: #3498db; color: white; padding: 2px 6px; border-radius: 3px 0 0 3px;',
  'background: #555; color: white; padding: 2px 6px; border-radius: 0 3px 3px 0;'
);
