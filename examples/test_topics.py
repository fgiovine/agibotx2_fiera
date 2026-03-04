#!/usr/bin/env python3
"""Test Topics - Elenca e monitora i topic attivi del robot."""

import subprocess
import sys


def main():
    print("=== Topic Robot Agibot X2 ===\n")

    result = subprocess.run(
        ['ros2', 'topic', 'list'],
        capture_output=True, text=True
    )

    topics = result.stdout.strip().split('\n')

    categories = {
        'Joint': [],
        'Camera': [],
        'LiDAR': [],
        'Audio': [],
        'MC': [],
        'Altro': [],
    }

    for t in topics:
        if 'joint' in t:
            categories['Joint'].append(t)
        elif 'sensor' in t and ('rgb' in t or 'depth' in t or 'stereo' in t):
            categories['Camera'].append(t)
        elif 'lidar' in t:
            categories['LiDAR'].append(t)
        elif 'audio' in t or 'tts' in t:
            categories['Audio'].append(t)
        elif '/aima/mc' in t:
            categories['MC'].append(t)
        else:
            categories['Altro'].append(t)

    for cat, topic_list in categories.items():
        if topic_list:
            print(f"--- {cat} ({len(topic_list)}) ---")
            for t in sorted(topic_list):
                print(f"  {t}")
            print()

    print(f"Totale: {len(topics)} topic")

    if len(sys.argv) > 1:
        topic = sys.argv[1]
        print(f"\n=== Echo {topic} ===")
        subprocess.run(['ros2', 'topic', 'echo', topic, '--once'])


if __name__ == '__main__':
    main()
