# Local overlay

Everything specific to one household lives here and stays out of git:
`.env` (HA address, SSH target), exported room and person config, trained
`.pkl` models, raw event logs.

```bash
cp local/.env.example local/.env   # then edit
scripts/deploy.sh                  # push the integration to your HA box
cd dashboard && docker compose --env-file ../local/.env up -d
```

Only this README and `.env.example` are tracked. See DEVELOPMENT.md.
