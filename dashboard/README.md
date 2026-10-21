# WhoLLM dashboards

![Standalone dashboard with demo data](../docs/assets/dashboard-live.png)

*Rendered from this code with demo data.*

This folder holds two ways to show WhoLLM on a screen. The full guide, with screenshots, is in
**[docs/dashboard.md](../docs/dashboard.md)**. There is also a single-file Lovelace card in
[`www/whollm-card.js`](../www/whollm-card.js).

| File | What it is |
|---|---|
| `index.html` | Standalone web dashboard (people, rooms, LLM analysis, activity timeline). |
| `docker-compose.yml` | Runs the dashboard behind nginx on port 3380 (`DASHBOARD_PORT` to change). |
| `nginx.conf` | Serves the page and proxies `/api/` and `/auth/token` to Home Assistant. |
| `whollm-dashboard.yaml` | A multi-section Lovelace dashboard (needs Mushroom and card-mod from HACS). |

## Standalone dashboard in three steps

1. Edit `nginx.conf` and replace every `192.168.0.233:8123` with your Home Assistant host and port.
   (The `HA_URL` variable in `docker-compose.yml` is **not** used by `nginx.conf`.)
2. `docker compose up -d`
3. Open `http://<your-host>:3380`, enter your Home Assistant address, and choose **Sign in with Home Assistant**.

It signs in with Home Assistant's OAuth flow. A long-lived access token is available under *Advanced*.

## Lovelace YAML dashboard

Copy `whollm-dashboard.yaml` into your Home Assistant config directory and register it:

```yaml
lovelace:
  dashboards:
    whollm:
      mode: yaml
      title: WhoLLM
      icon: mdi:brain
      show_in_sidebar: true
      filename: whollm-dashboard.yaml
```

The file uses demo entity names (`sensor.alice_room`, and so on). Replace them with yours.

## Customising the standalone page

Room tiles come from a fixed list near the top of the script in `index.html`:

```javascript
rooms: ['office', 'living_room', 'bedroom', 'kitchen', 'bathroom', 'entry', 'away'],
roomIcons: { office: '🖥️', living_room: '🛋️', /* ... */ }
```

Add your own rooms and icons there. Colours are CSS variables in the `:root` block (`--accent-primary`,
`--room-office`, and so on).

## Known limitations

- The vision panel and *Identify Now* button still use the old `llm_presence` integration name and do not work yet.
  Use the `whollm.identify_person` service from Home Assistant instead.
- The activity timeline is built in the browser from changes it sees while open; it is not history.

More in [Status & roadmap](../docs/status.md#known-issues) and [Troubleshooting](../docs/troubleshooting.md#dashboard-and-card-problems).

## License

MIT, as part of [WhoLLM](https://github.com/owens-ben/WhoLLM).
