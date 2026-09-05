"""Build benchmark games from the raw WordNet triples.

Sampling categories are grouped by taxonomic distance using the terminology
from the paper: close (at most four edges) and deep (more than four edges).

Includes optional quality filtering to remove:
- Instances where target hypothesis appears in examples
- Overly specific examples (>4 words)
"""

import json
from typing import List, Dict, Any
from pathlib import Path


def _filter_G_standalone(instance: Dict) -> bool:
    """
    Filter if G appears as a standalone example.
    Returns True if instance should be FILTERED OUT.
    """
    G = instance['target_hypothesis']
    examples = instance['initial_examples']

    # Normalize G
    G_normalized = G.lower().replace('_', ' ')

    # Check if G appears as standalone item
    for ex in examples:
        ex_normalized = ex.lower().strip()
        if ex_normalized == G_normalized:
            return True  # FILTER OUT

    return False  # KEEP


def _filter_G_in_examples(instance: Dict) -> bool:
    """
    Filter if G appears meaningfully in examples.
    Avoids false positives from compounds like 'Golgi body'.
    Returns True if instance should be FILTERED OUT.
    """
    G = instance['target_hypothesis']
    examples = instance['initial_examples']

    G_tokens = set(G.lower().replace('_', ' ').split())

    # Special handling for compound-prone words
    compound_exceptions = {
        'body': ['golgi body', 'carotid body', 'ciliary body', 'vitreous body',
                'cell body', 'polar body', 'inclusion body', 'mamillary body'],
        'part': ['mouthpart', 'counterpart', 'spare part'],
        'plant': ['eggplant', 'transplant', 'implant'],
        'food': ['seafood', 'dog food', 'cat food'],
        'work': ['artwork', 'network', 'framework', 'handiwork'],
    }

    examples_text = ' '.join(examples).lower()

    for token in G_tokens:
        if len(token) <= 3:
            continue  # Skip short words

        if token in examples_text:
            # Check if it's a known compound exception
            if token in compound_exceptions:
                is_compound = any(compound in examples_text
                                for compound in compound_exceptions[token])
                if is_compound:
                    continue  # Likely compound word, allow it

            # If we get here, G token appears and it's not a compound
            return True  # FILTER OUT

    return False  # KEEP


def _filter_overly_specific(instance: Dict) -> bool:
    """
    Filter instances with examples containing more than 4 words.
    Returns True if instance should be FILTERED OUT.
    """
    examples = instance['initial_examples']

    for ex in examples:
        if len(ex.split()) > 4:
            return True  # FILTER OUT

    return False  # KEEP


class GameDatasetBuilder:
    """
    Build game instances and group sampling categories by taxonomic distance.

    Distance groups:
    - Close: distance <= 4
    - Deep: distance > 4
    """

    def __init__(self, data_path: str = "raw_data/wordnet_triples.json", apply_filters: bool = True):
        """
        Initialize the stratifier.

        Args:
            data_path: Path to raw data file
            apply_filters: If True, apply quality filters during instance creation
        """
        self.data_path = Path(data_path)
        self.raw_data = None
        self.game_instances = []
        self.apply_filters = apply_filters
        self.filter_stats = {
            'total_created': 0,
            'filtered_G_standalone': 0,
            'filtered_G_in_examples': 0,
            'filtered_overly_specific': 0,
            'total_filtered': 0,
            'total_kept': 0
        }

    def load_data(self) -> List[Dict[str, Any]]:
        """Load the raw dataset from JSON."""
        with open(self.data_path, 'r') as f:
            self.raw_data = json.load(f)
        return self.raw_data

    def calculate_distance_group(self, hypernym_chain: List[str]) -> tuple[str, int]:
        """
        Calculate the distance group from the hypernym chain length.

        Args:
            hypernym_chain: List of hypernyms from specific to general

        Returns:
            Tuple of (distance_group_label, distance)
        """
        distance = len(hypernym_chain)

        if distance <= 4:
            return "close", distance
        else:
            return "deep", distance

    def create_game_instances(self) -> List[Dict[str, Any]]:
        """
        Transform raw data into game instances with taxonomic-distance groups.

        Optionally applies quality filters to remove:
        - Instances where target hypothesis appears in examples
        - Overly specific examples (>4 words)

        Each instance contains:
        - sampling_hypothesis (S): First element of hypernyms
        - target_hypothesis (G): Last element of hypernyms
        - initial_examples (E): The three triplet items
        - hypernym_path: Full taxonomy path
        - distance_group: ``close`` or ``deep``
        - distance: Numeric distance measure
        """
        if self.raw_data is None:
            self.load_data()

        self.game_instances = []
        self.filter_stats = {
            'total_created': 0,
            'filtered_G_standalone': 0,
            'filtered_G_in_examples': 0,
            'filtered_overly_specific': 0,
            'total_filtered': 0,
            'total_kept': 0
        }

        for idx, item in enumerate(self.raw_data):
            hypernyms = item.get("hypernyms", [])

            # Skip if we have only 2 properties or less (not enough for a meaningful distance)
            if len(hypernyms) < 3:
                continue

            sampling_hypothesis = hypernyms[0]  # Most specific (S)
            target_hypothesis = hypernyms[-1]   # Most general (G)

            distance_group, distance = self.calculate_distance_group(hypernyms)

            game_instance = {
                "game_id": idx,
                "sampling_hypothesis": sampling_hypothesis,
                "target_hypothesis": target_hypothesis,
                "initial_examples": item["triplets"],
                "hypernym_path": hypernyms,
                "hypernym_synsets": item.get("hypernyms_synsets", []),
                "distance_group": distance_group,
                "distance": distance,
                "psc": item.get("PSC", ""),
                "psc_synset": item.get("PSC_synset", ""),
                "psc_definitions": item.get("PSC_definitions", [])
            }

            self.filter_stats['total_created'] += 1

            # Apply quality filters if enabled
            if self.apply_filters:
                should_filter = False

                # Filter 1: G as standalone example
                if _filter_G_standalone(game_instance):
                    self.filter_stats['filtered_G_standalone'] += 1
                    should_filter = True

                # Filter 2: G tokens in examples (if not already filtered)
                if not should_filter and _filter_G_in_examples(game_instance):
                    self.filter_stats['filtered_G_in_examples'] += 1
                    should_filter = True

                # Filter 3: Overly specific examples
                if not should_filter and _filter_overly_specific(game_instance):
                    self.filter_stats['filtered_overly_specific'] += 1
                    should_filter = True

                if should_filter:
                    self.filter_stats['total_filtered'] += 1
                    continue  # Skip this instance

            self.filter_stats['total_kept'] += 1
            self.game_instances.append(game_instance)

        return self.game_instances

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics including filtering info."""
        if not self.game_instances:
            self.create_game_instances()

        total = len(self.game_instances)
        close = sum(1 for g in self.game_instances if g["distance_group"] == "close")
        deep = sum(1 for g in self.game_instances if g["distance_group"] == "deep")

        stats = {
            "total_instances": total,
            "close": close,
            "deep": deep,
            "close_pct": round(100 * close / total, 2) if total > 0 else 0,
            "deep_pct": round(100 * deep / total, 2) if total > 0 else 0,
        }

        # Add filter statistics if filtering was applied
        if self.apply_filters:
            stats['filtering'] = {
                'enabled': True,
                'total_created': self.filter_stats['total_created'],
                'total_filtered': self.filter_stats['total_filtered'],
                'total_kept': self.filter_stats['total_kept'],
                'filter_rate': round(100 * self.filter_stats['total_filtered'] /
                                   self.filter_stats['total_created'], 2)
                              if self.filter_stats['total_created'] > 0 else 0,
                'reasons': {
                    'G_standalone': self.filter_stats['filtered_G_standalone'],
                    'G_in_examples': self.filter_stats['filtered_G_in_examples'],
                    'overly_specific': self.filter_stats['filtered_overly_specific']
                }
            }
        else:
            stats['filtering'] = {'enabled': False}

        return stats

    def save_dataset(self, output_path: str = "data/game_instances.json"):
        """Save processed game instances to JSON."""
        if not self.game_instances:
            self.create_game_instances()

        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(self.game_instances, f, indent=2)

        print(f"Saved {len(self.game_instances)} game instances to {output_path}")

    def save_by_distance_group(self, output_dir: str = "data"):
        """Save separate datasets for the close and deep groups."""
        if not self.game_instances:
            self.create_game_instances()

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        for distance_group in ["close", "deep"]:
            instances = [g for g in self.game_instances if g["distance_group"] == distance_group]
            filepath = output_dir / f"game_instances_{distance_group}.json"

            with open(filepath, 'w') as f:
                json.dump(instances, f, indent=2)

            print(f"Saved {len(instances)} {distance_group} instances to {filepath}")


if __name__ == "__main__":
    # Example usage
    stratifier = GameDatasetBuilder("raw_data/wordnet_triples.json", apply_filters=True)
    stratifier.load_data()
    instances = stratifier.create_game_instances()

    # Print statistics
    stats = stratifier.get_statistics()
    print("\nDataset Statistics:")
    print(f"Total instances: {stats['total_instances']}")
    print(f"Close: {stats['close']} ({stats['close_pct']}%)")
    print(f"Deep: {stats['deep']} ({stats['deep_pct']}%)")

    # Print filtering statistics if enabled
    if stats['filtering']['enabled']:
        print("\nQuality Filtering:")
        print(f"  Total created: {stats['filtering']['total_created']}")
        print(f"  Filtered out: {stats['filtering']['total_filtered']} ({stats['filtering']['filter_rate']}%)")
        print(f"  Kept: {stats['filtering']['total_kept']}")
        print(f"  Reasons:")
        print(f"    - G standalone: {stats['filtering']['reasons']['G_standalone']}")
        print(f"    - G in examples: {stats['filtering']['reasons']['G_in_examples']}")
        print(f"    - Overly specific: {stats['filtering']['reasons']['overly_specific']}")

    # Show a sample instance
    if instances:
        print("\nSample Game Instance:")
        print(json.dumps(instances[0], indent=2))

    # Save datasets
    stratifier.save_dataset()
    stratifier.save_by_distance_group()
