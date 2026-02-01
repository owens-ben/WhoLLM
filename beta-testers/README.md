# WhoLLM Beta Tester Agents

This directory contains 5 simulated beta tester profiles with varied home setups for comprehensive testing.

## Tester Profiles

| Agent | Setup | Focus Area |
|-------|-------|------------|
| Alice | Studio apartment, single person | Basic functionality |
| Bob | Family home, 4 people + pet | Multi-person tracking |
| Carol | Smart home enthusiast | Dense sensors, cameras |
| Dan | Minimal setup | Edge cases, fallbacks |
| Eve | Remote worker | Device tracker, away detection |

## Usage

```bash
# Run all beta tester scenarios
python beta-testers/run_all_testers.py

# Run a specific tester
python beta-testers/run_all_testers.py --tester alice
```

## Test Coverage

Each tester validates:
- Basic room detection
- Confidence scoring
- Edge case handling
- Device tracker integration (Eve)
- Multi-person scenarios (Bob)
- Camera AI detection (Carol)
