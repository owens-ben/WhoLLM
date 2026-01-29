# WhoLLM Examples

This directory contains example configurations for using the WhoLLM integration.

## Files

- `example_automation.yaml` - Example automations using WhoLLM sensors

## Usage

Copy the relevant sections from `example_automation.yaml` into your Home Assistant `configuration.yaml` or create separate automation files in your `automations/` directory.

## Tips

1. **Combine with other sensors**: Use LLM presence sensors alongside motion sensors in Bayesian sensors for higher accuracy
2. **Use confidence levels**: Check the `confidence` attribute to only trigger automations when confidence is high
3. **Add delays**: When turning off lights, add a delay to account for brief absences
4. **Room-specific logic**: Create different automations for different rooms based on typical usage patterns

