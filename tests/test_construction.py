import json

from src.benchmark.construction import GameDatasetBuilder


def test_game_dataset_builder_uses_paper_distance_groups(tmp_path):
    raw_path = tmp_path / "wordnet_triples.json"
    raw_path.write_text(
        json.dumps(
            [
                {"hypernyms": ["feline", "carnivore", "mammal", "animal"], "triplets": ["cat", "lion", "tiger"]},
                {"hypernyms": ["cat", "feline", "carnivore", "mammal", "vertebrate", "animal"], "triplets": ["tabby", "tom", "kitten"]},
            ]
        ),
        encoding="utf-8",
    )

    builder = GameDatasetBuilder(str(raw_path), apply_filters=False)
    instances = builder.create_game_instances()

    assert [instance["distance_group"] for instance in instances] == ["close", "deep"]
    assert builder.get_statistics() == {
        "total_instances": 2,
        "close": 1,
        "deep": 1,
        "close_pct": 50.0,
        "deep_pct": 50.0,
        "filtering": {"enabled": False},
    }
