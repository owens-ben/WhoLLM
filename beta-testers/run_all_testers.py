#!/usr/bin/env python3
"""
WhoLLM Beta Tester Runner

Executes all beta tester scenarios against a WhoLLM instance.

Usage:
    python run_all_testers.py --ha-url http://localhost:8123 --ha-token TOKEN
    python run_all_testers.py --tester alice  # Run specific tester
    python run_all_testers.py --list  # List available testers
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
import yaml


TESTERS_DIR = Path(__file__).parent / "testers"


@dataclass
class TestResult:
    """Result of a single test scenario."""
    tester: str
    scenario: str
    passed: bool
    expected: dict
    actual: dict
    confidence: float
    message: str = ""


@dataclass  
class TesterReport:
    """Report for a complete beta tester run."""
    tester: str
    profile: dict
    total_scenarios: int
    passed: int
    failed: int
    results: list[TestResult] = field(default_factory=list)
    avg_confidence: float = 0.0


class BetaTesterRunner:
    """Run beta tester scenarios against WhoLLM."""

    def __init__(self, ha_url: str, ha_token: str):
        self.ha_url = ha_url.rstrip("/")
        self.ha_token = ha_token
        self.headers = {
            "Authorization": f"Bearer {ha_token}",
            "Content-Type": "application/json",
        }

    async def get_state(self, entity_id: str) -> dict | None:
        """Get entity state from Home Assistant."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.ha_url}/api/states/{entity_id}",
                headers=self.headers,
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return None

    async def set_state(self, entity_id: str, state: str) -> bool:
        """Set entity state in Home Assistant."""
        domain = entity_id.split(".")[0]
        
        if state.lower() in ("on", "home", "playing"):
            service = "turn_on"
        else:
            service = "turn_off"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.ha_url}/api/services/{domain}/{service}",
                headers=self.headers,
                json={"entity_id": entity_id},
            ) as resp:
                return resp.status == 200

    async def run_scenario(
        self,
        tester_name: str,
        scenario: dict,
        persons: list[dict],
    ) -> TestResult:
        """Run a single test scenario."""
        scenario_name = scenario.get("name", "Unknown")
        expected = scenario.get("expected", {})
        
        print(f"    Running: {scenario_name}")
        
        # Execute steps
        for step in scenario.get("steps", []):
            if "set" in step:
                # Parse entity settings
                settings = step["set"]
                for setting in settings.split(","):
                    setting = setting.strip()
                    if "=" in setting:
                        entity_id, state = setting.split("=")
                        await self.set_state(entity_id.strip(), state.strip())
            
            if "wait" in step:
                await asyncio.sleep(step["wait"])
        
        # Check results
        actual = {}
        passed = True
        
        for person in persons:
            person_name = person.get("name", "").lower()
            sensor_id = f"sensor.{person_name}_room"
            
            state = await self.get_state(sensor_id)
            if state:
                actual[f"{person_name}_room"] = state.get("state")
                actual[f"{person_name}_confidence"] = state.get("attributes", {}).get("confidence", 0)
                
                # Check expected room
                expected_key = f"{person_name}_room"
                if expected_key in expected:
                    if actual[f"{person_name}_room"] != expected[expected_key]:
                        passed = False
                
                # Check confidence bounds
                if "min_confidence" in expected:
                    if actual.get(f"{person_name}_confidence", 0) < expected["min_confidence"]:
                        passed = False
        
        confidence = actual.get(f"{persons[0]['name'].lower()}_confidence", 0) if persons else 0
        
        return TestResult(
            tester=tester_name,
            scenario=scenario_name,
            passed=passed,
            expected=expected,
            actual=actual,
            confidence=confidence,
            message="PASS" if passed else "FAIL",
        )

    async def run_tester(self, tester_file: Path) -> TesterReport:
        """Run all scenarios for a beta tester."""
        with open(tester_file) as f:
            config = yaml.safe_load(f)
        
        tester_name = config["profile"]["name"]
        print(f"\n{'='*60}")
        print(f"Running Beta Tester: {tester_name}")
        print(f"Focus: {config['profile']['focus']}")
        print(f"{'='*60}")
        
        persons = config["home_config"]["persons"]
        scenarios = config.get("test_scenarios", [])
        
        results = []
        for scenario in scenarios:
            result = await self.run_scenario(tester_name, scenario, persons)
            results.append(result)
            status = "✓" if result.passed else "✗"
            print(f"      {status} {result.scenario}")
        
        passed = sum(1 for r in results if r.passed)
        avg_conf = sum(r.confidence for r in results) / len(results) if results else 0
        
        return TesterReport(
            tester=tester_name,
            profile=config["profile"],
            total_scenarios=len(scenarios),
            passed=passed,
            failed=len(scenarios) - passed,
            results=results,
            avg_confidence=avg_conf,
        )


def list_testers():
    """List available beta testers."""
    print("\nAvailable Beta Testers:")
    print("-" * 40)
    
    for tester_file in sorted(TESTERS_DIR.glob("*.yaml")):
        with open(tester_file) as f:
            config = yaml.safe_load(f)
        
        name = config["profile"]["name"]
        focus = config["profile"]["focus"]
        print(f"  {tester_file.stem}: {name} ({focus})")


def generate_report(reports: list[TesterReport], output_path: Path):
    """Generate a markdown test report."""
    total_tests = sum(r.total_scenarios for r in reports)
    total_passed = sum(r.passed for r in reports)
    
    report = f"""# WhoLLM Beta Tester Report

**Generated**: {datetime.now().isoformat()}

## Summary

| Metric | Value |
|--------|-------|
| Testers Run | {len(reports)} |
| Total Scenarios | {total_tests} |
| Passed | {total_passed} |
| Failed | {total_tests - total_passed} |
| Pass Rate | {total_passed/total_tests*100:.1f}% |

## Results by Tester

"""
    
    for r in reports:
        status = "✓ PASS" if r.passed >= r.total_scenarios * 0.5 else "✗ FAIL"
        report += f"""### {r.tester}
**Focus**: {r.profile.get('focus', 'N/A')}  
**Result**: {r.passed}/{r.total_scenarios} ({status})  
**Avg Confidence**: {r.avg_confidence:.1%}

| Scenario | Status | Confidence |
|----------|--------|------------|
"""
        for result in r.results:
            status = "PASS" if result.passed else "FAIL"
            report += f"| {result.scenario} | {status} | {result.confidence:.1%} |\n"
        
        report += "\n"
    
    output_path.write_text(report)
    print(f"\nReport saved to: {output_path}")


async def main():
    parser = argparse.ArgumentParser(description="WhoLLM Beta Tester Runner")
    parser.add_argument("--ha-url", default="http://localhost:8123", help="Home Assistant URL")
    parser.add_argument("--ha-token", default="", help="HA long-lived access token")
    parser.add_argument("--tester", help="Run specific tester (e.g., alice, bob)")
    parser.add_argument("--list", action="store_true", help="List available testers")
    parser.add_argument("--output", type=Path, default=Path("beta-test-report.md"), help="Output report path")
    
    args = parser.parse_args()
    
    if args.list:
        list_testers()
        return
    
    if not args.ha_token:
        print("Error: --ha-token is required")
        print("Create a long-lived access token in HA: Profile -> Security -> Long-lived access tokens")
        sys.exit(1)
    
    runner = BetaTesterRunner(args.ha_url, args.ha_token)
    
    # Find tester files
    if args.tester:
        tester_files = list(TESTERS_DIR.glob(f"{args.tester}*.yaml"))
        if not tester_files:
            print(f"Tester not found: {args.tester}")
            list_testers()
            sys.exit(1)
    else:
        tester_files = sorted(TESTERS_DIR.glob("*.yaml"))
    
    # Run testers
    reports = []
    for tester_file in tester_files:
        report = await runner.run_tester(tester_file)
        reports.append(report)
    
    # Generate report
    generate_report(reports, args.output)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in reports:
        status = "✓" if r.passed >= r.total_scenarios * 0.5 else "✗"
        print(f"  {status} {r.tester}: {r.passed}/{r.total_scenarios} passed")


if __name__ == "__main__":
    asyncio.run(main())
