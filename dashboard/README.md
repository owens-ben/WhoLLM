# LLM Room Presence Dashboard

A beautiful, modern dashboard for visualizing LLM-powered room presence detection.

![Dashboard Preview](preview.png)

## Features

- 🧠 Real-time presence visualization
- 👥 Multi-person and pet tracking
- 🏠 Room occupancy overview with visual indicators
- 📊 Confidence level display
- 🤖 LLM response viewer
- 📜 Activity timeline
- 🌙 Dark theme with animated gradients
- 📱 Fully responsive design

## Installation Options

### Option 1: Docker (Recommended)

```bash
cd llm-room-presence/dashboard
docker compose up -d
```

Access at: `http://YOUR_HOST:3380`

### Option 2: Home Assistant Dashboard

Import `llm-presence-dashboard.yaml` into Home Assistant:

1. Copy `llm-presence-dashboard.yaml` to your HA config directory
2. Add to `configuration.yaml`:

```yaml
lovelace:
  mode: yaml
  dashboards:
    llm-presence:
      mode: yaml
      title: LLM Presence
      icon: mdi:brain
      show_in_sidebar: true
      filename: llm-presence-dashboard.yaml
```

3. Restart Home Assistant

### Option 3: Static File Server

Serve the `index.html` file with any web server (nginx, Apache, Python http.server).

```bash
# Quick test with Python
cd llm-room-presence/dashboard
python3 -m http.server 3380
```

## Configuration

On first load, enter your Home Assistant details:

1. **Home Assistant URL**: `http://homeassistant.local:8123` (or your HA IP/hostname)
2. **Long-Lived Access Token**: Generate in HA → Profile → Long-Lived Access Tokens

Settings are saved to browser localStorage.

## Requirements

- LLM Room Presence integration installed in Home Assistant
- Home Assistant accessible from the browser
- Valid long-lived access token

## Customization

### Adding Rooms

Edit the `CONFIG.rooms` array in `index.html`:

```javascript
rooms: ['office', 'living_room', 'bedroom', 'kitchen', 'bathroom', 'garage', 'away'],
roomIcons: {
    office: '🖥️',
    living_room: '🛋️',
    bedroom: '🛏️',
    kitchen: '🍳',
    bathroom: '🚿',
    garage: '🚗',
    away: '🚪',
    unknown: '❓'
}
```

### Changing Colors

Modify CSS variables in the `:root` section:

```css
:root {
    --accent-primary: #6366f1;    /* Main accent color */
    --accent-success: #10b981;    /* Success/occupied color */
    --room-office: #3b82f6;       /* Office room color */
    /* ... etc */
}
```

## Files

| File | Description |
|------|-------------|
| `index.html` | Standalone web dashboard |
| `llm-presence-dashboard.yaml` | Home Assistant Lovelace dashboard |
| `docker-compose.yml` | Docker deployment |
| `nginx.conf` | Nginx configuration |

## Integration with Homepage

The Docker container includes Homepage labels for automatic discovery:

- **Group**: AI
- **Name**: LLM Presence Dashboard
- **Port**: 3380

## Troubleshooting

### "Waiting for LLM Presence data..."

- Ensure the LLM Room Presence integration is configured in Home Assistant
- Check that entities like `sensor.{person_name}_room` exist (e.g., `sensor.alice_room`)
- Verify the integration is polling (check HA logs)

### Connection Failed

- Verify Home Assistant URL is correct
- Check the access token is valid
- Ensure CORS is not blocking requests (HA should allow API access)

### No Room Data

- The integration needs to be configured with persons/pets and rooms
- Go to Settings → Devices & Services → LLM Room Presence → Configure

## License

MIT License - Part of the [LLM Room Presence](https://github.com/owens-ben/llm-room-presence) project.



