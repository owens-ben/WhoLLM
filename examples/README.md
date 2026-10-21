# WhoLLM examples

Copy-paste automation YAML for WhoLLM. More recipes, with explanations, are in
**[docs/automations.md](../docs/automations.md)**.

## Files

| File | What it contains |
|---|---|
| [`example_automation.yaml`](example_automation.yaml) | Lights on and off by room, a pet alert, and a Bayesian sensor that fuses WhoLLM with a motion sensor. |
| [`../automations/vision_on_motion.yaml`](../automations/vision_on_motion.yaml) | Run camera identification when a camera sees a person or animal. |

## Usage

Copy the parts you want into your Home Assistant `configuration.yaml`, or into the automation editor's YAML mode.
Change the entity IDs (`binary_sensor.alice_in_office`, `light.office_lights`, and so on) to match your own.

## Tips

1. **Combine with other sensors.** Use WhoLLM's room sensors alongside motion sensors in Bayesian sensors for higher
   accuracy.
2. **Use the confidence attribute.** `confidence` runs from 0 to 1. Guard automations with something like
   `{{ state_attr('sensor.alice_room', 'confidence') | float(0) >= 0.6 }}`.
3. **Add delays.** Use `for:` on triggers, or a delay before turning lights off, to allow for brief absences.
4. **Room-specific logic.** Use different automations for different rooms, based on how each is used.
