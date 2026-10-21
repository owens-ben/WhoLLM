# ML & training data

[← Development](development.md) · [Guide home](README.md)

WhoLLM quietly records what it decides, so you can train your own models on your own household later. This page
describes the recording and the helper scripts.

> [!WARNING]
> **Experimental.** The training scripts and the ML predictor module are not connected to the running integration, so
> trained models do not change WhoLLM's answers yet (see [Status](status.md)). The scripts also need small edits
> (listed below) before they work for your home.

## The data you are already collecting

Every poll writes one JSON line per person or pet to `/config/whollm_events.jsonl` (30 days or 100 MB by default).
Each record has the timestamp, person, room, confidence, the LLM's raw answer, the evidence, and summaries of the time
and of which sensors were active. See [What gets logged](how-it-works.md#what-gets-logged).

Because the rooms come from WhoLLM's own weighted decisions, the log is only as good as your sensor mapping. Confident
records (say 70 % and up) make better training labels than low-confidence ones.

## The scripts

All live in [`scripts/`](../scripts/). They need `numpy` and `scikit-learn`, which are not part of the integration's
requirements.

| Script | What it does |
|---|---|
| `sync_training_data.sh [HA_CONFIG]` | Copies the event log from your Home Assistant config directory into `data/ha_events.jsonl`, then runs `prepare_ml_training_data.py`. |
| `prepare_ml_training_data.py -i events.jsonl -o OUT_DIR [--val-split 0.2] [--min-confidence 0.0]` | Turns events into features and labels, splits into `train.jsonl` and `val.jsonl`, and writes `metadata.json`. |
| `train_model.py` | Trains several classifiers on the prepared data and saves the best to `models/presence_model.pkl`. |
| `train_per_person_models.py` | Trains one calibrated model per person, with uncertainty estimates, and saves `models/per_person_models.pkl`. |
| `extract_training_data.py -i logbook.json -o events.jsonl [-a]` | Builds training records from a Home Assistant **logbook** export, to bootstrap before you have your own event log. |
| `merge_logbook_data.py [files…]` | Merges and de-duplicates several logbook exports from `data/raw/`. |

A typical run:

```bash
pip install numpy scikit-learn
./scripts/sync_training_data.sh /path/to/ha/config     # see "Before you run" first
python scripts/train_per_person_models.py
```

`data/` and `models/` are git-ignored on purpose, because they contain personal data.

### Before you run them

These are known rough edges (also listed in [Status](status.md#known-issues)):

- `sync_training_data.sh` looks for **`llm_presence_events.jsonl`**. The integration writes `whollm_events.jsonl`, so
  point the script at that name or copy the file by hand to `data/ha_events.jsonl`.
- `prepare_ml_training_data.py` has the room list and people (`Alice`, `Bob`, `Max`) written into it. Edit them to match
  your household.
- `extract_training_data.py` has `PEOPLE = []`, which it auto-detects, and a keyword table for matching entity names
  to rooms.

## The predictor module

[`custom_components/whollm/ml_predictor.py`](../custom_components/whollm/ml_predictor.py) can load
`per_person_models.pkl` from `/config/custom_components/whollm/models/` and predict a room, and flags a prediction as
uncertain when the top probability is below 0.4 (with a 15-minute cooldown between nudges). Nothing calls it yet.

> [!CAUTION]
> Models are stored with Python's `pickle`, which can run code when loaded. Only load model files **you trained
> yourself**, and never ones from someone else.

## Longer-term ideas

[`ML_ENHANCEMENT_PROPOSAL.md`](ML_ENHANCEMENT_PROPOSAL.md) is a design proposal covering face recognition, an image
queue and a presence database. It is an ideas document, not a description of shipped features.
