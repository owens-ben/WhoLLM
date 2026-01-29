#!/usr/bin/env python3
"""
Merge multiple logbook JSON files into a single training dataset.
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def merge_logbook_files(input_files: list[Path], output_file: Path):
    """Merge multiple logbook JSON files."""
    all_entries = []
    seen_keys = set()  # Deduplicate based on entity_id + when
    
    for input_file in input_files:
        print(f"Processing {input_file}...")
        try:
            with open(input_file) as f:
                data = json.load(f)
            
            entries = data.get('entries', []) if isinstance(data, dict) else data
            
            for entry in entries:
                # Create unique key for deduplication
                key = (entry.get('entity_id', ''), entry.get('when', ''))
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_entries.append(entry)
                    
        except Exception as e:
            print(f"  Error: {e}")
    
    # Sort by timestamp
    all_entries.sort(key=lambda x: x.get('when', ''))
    
    print(f"\nTotal unique entries: {len(all_entries)}")
    
    # Write merged output
    output_data = {
        'merged_at': datetime.now().isoformat(),
        'source_files': [str(f) for f in input_files],
        'total_entries': len(all_entries),
        'entries': all_entries,
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Written to {output_file}")
    
    # Summary
    entities = {}
    for e in all_entries:
        eid = e.get('entity_id', 'unknown')
        domain = eid.split('.')[0] if '.' in eid else 'unknown'
        entities[domain] = entities.get(domain, 0) + 1
    
    print("\nEntries by domain:")
    for domain, count in sorted(entities.items(), key=lambda x: -x[1]):
        print(f"  {domain}: {count}")


if __name__ == '__main__':
    # Use relative paths from script location
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    raw_dir = project_dir / 'data' / 'raw'
    
    input_files = []
    
    # Add raw directory files
    if raw_dir.exists():
        input_files.extend(raw_dir.glob('*.json'))
    
    # Allow passing additional files as arguments
    for arg in sys.argv[1:]:
        input_path = Path(arg)
        if input_path.exists():
            input_files.append(input_path)
    
    if not input_files:
        print("No input files found!")
        print(f"Place logbook JSON files in: {raw_dir}")
        print("Or pass file paths as arguments")
        sys.exit(1)
    
    print(f"Found {len(input_files)} input files")
    
    output_file = raw_dir / 'merged_logbook.json'
    merge_logbook_files(input_files, output_file)
