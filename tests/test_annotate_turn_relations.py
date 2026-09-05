import json
from pathlib import Path

from src.annotation.annotate_relations import (
    annotate_many,
    annotate_results_file,
    find_results_files,
    iter_test_turn_entries,
    TurnRelationAnnotator,
)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCreate:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self.content)


class _SequenceFakeCreate:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not self.contents:
            raise AssertionError("No fake response left")
        return _FakeResponse(self.contents.pop(0))


class _FakeChatCompletions:
    def __init__(self, create_fn):
        self.create = create_fn


class _FakeChat:
    def __init__(self, create_fn):
        self.completions = _FakeChatCompletions(create_fn)


class _FakeClient:
    def __init__(self, create_fn):
        self.chat = _FakeChat(create_fn)


def test_find_results_files_respects_recursive_flag(tmp_path):
    results_root = tmp_path / "results"
    (results_root / "model-a" / "game_results.json").parent.mkdir(parents=True)
    (results_root / "model-a" / "game_results.json").write_text("[]", encoding="utf-8")
    (results_root / "model-b" / "game_results.json").parent.mkdir(parents=True)
    (results_root / "model-b" / "game_results.json").write_text("[]", encoding="utf-8")
    (results_root / "archived" / "model-c" / "game_results.json").parent.mkdir(parents=True)
    (results_root / "archived" / "model-c" / "game_results.json").write_text("[]", encoding="utf-8")

    recursive = find_results_files(results_root, recursive=True)
    non_recursive = find_results_files(results_root, recursive=False)

    assert len(recursive) == 3
    assert len(non_recursive) == 2


def test_iter_test_turn_entries_skips_guess_turns():
    results = [
        {
            "game_id": 1,
            "target_property": "animal",
            "sampling_hypothesis": "fish",
            "distance_group": "close",
            "distance": 3,
            "turns": [
                {
                    "turn_number": 1,
                    "action": {"action": "test", "hypothesis": "mammal"},
                    "player_reasoning": "test",
                    "oracle_response": {"feedback": "ok"},
                },
                {
                    "turn_number": 2,
                    "action": {"action": "guess", "property": "animal"},
                },
            ],
        }
    ]

    entries = list(iter_test_turn_entries(results))

    assert len(entries) == 1
    assert entries[0]["turn_number"] == 1
    assert entries[0]["hypothesis"] == "mammal"
    assert "target_property" in entries[0]


def test_annotate_relation_accepts_valid_relation(monkeypatch):
    create_fn = _FakeCreate(
        '{"relation": "hypothesis_included_in_target", "explanation": "Mammals are animals."}'
    )

    def fake_create_provider_client(provider, model, api_key=None):
        return {
            "api_key": "test-key",
            "client": _FakeClient(create_fn),
            "api_version": None,
            "azure_endpoint": None,
        }

    monkeypatch.setattr(
        "src.annotation.annotate_relations._create_provider_client",
        fake_create_provider_client,
    )

    annotator = TurnRelationAnnotator(model="gpt-5-mini")
    annotation = annotator.annotate_relation(target_property="animal", hypothesis="mammal")

    assert annotation["status"] == "ok"
    assert annotation["relation"] == "hypothesis_included_in_target"
    assert len(create_fn.calls) == 1
    assert create_fn.calls[0]["reasoning"] == {"effort": "minimal"}


def test_annotate_relation_rejects_invalid_relation(monkeypatch):
    def fake_create_provider_client(provider, model, api_key=None):
        return {
            "api_key": "test-key",
            "client": _FakeClient(_FakeCreate('{"relation": "overlaps", "explanation": "bad label"}')),
            "api_version": None,
            "azure_endpoint": None,
        }

    monkeypatch.setattr(
        "src.annotation.annotate_relations._create_provider_client",
        fake_create_provider_client,
    )

    annotator = TurnRelationAnnotator(model="gpt-5-mini")
    annotation = annotator.annotate_relation(target_property="animal", hypothesis="carnivore")

    assert annotation["status"] == "annotation_error"
    assert annotation["relation"] is None
    assert "Invalid relation label" in annotation["error"]


def test_annotate_relation_retries_with_smaller_contract_after_invalid_json(monkeypatch):
    create_fn = _SequenceFakeCreate(
        [
            '{"relation": "hypothesis_included_in_target",',
            '{"relation": "hypothesis_included_in_target"}',
        ]
    )

    def fake_create_provider_client(provider, model, api_key=None):
        return {
            "api_key": "test-key",
            "client": _FakeClient(create_fn),
            "api_version": None,
            "azure_endpoint": None,
        }

    monkeypatch.setattr(
        "src.annotation.annotate_relations._create_provider_client",
        fake_create_provider_client,
    )

    annotator = TurnRelationAnnotator(model="gpt-5-mini")
    annotation = annotator.annotate_relation(target_property="animal", hypothesis="mammal")

    assert annotation["status"] == "ok"
    assert annotation["relation"] == "hypothesis_included_in_target"
    assert annotation["explanation"] is None
    assert len(create_fn.calls) == 2


def test_annotate_results_file_writes_output_and_continues_on_errors(tmp_path):
    results_file = tmp_path / "model-x" / "game_results.json"
    results_file.parent.mkdir(parents=True)
    results_file.write_text(
        json.dumps(
            [
                {
                    "game_id": 7,
                    "target_property": "animal",
                    "sampling_hypothesis": "fish",
                    "distance_group": "deep",
                    "distance": 5,
                    "turns": [
                        {
                            "turn_number": 1,
                            "action": {"action": "test", "hypothesis": "mammal"},
                            "oracle_response": {"feedback": "one"},
                        },
                        {
                            "turn_number": 2,
                            "action": {"action": "test", "hypothesis": "broken"},
                            "oracle_response": {"feedback": "two"},
                        },
                        {
                            "turn_number": 3,
                            "action": {"action": "guess", "property": "animal"},
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    class _FakeAnnotator:
        model = "gpt-5-mini"
        provider = "openai"

        def annotate_relation(self, target_property, hypothesis):
            if hypothesis == "broken":
                return {
                    "status": "annotation_error",
                    "relation": None,
                    "explanation": None,
                    "raw_response": "{}",
                    "error": "bad response",
                }
            return {
                "status": "ok",
                "relation": "hypothesis_included_in_target",
                "explanation": "ok",
                "raw_response": "{}",
                "error": None,
            }

    summary = annotate_results_file(results_file=results_file, annotator=_FakeAnnotator())
    output_path = results_file.parent / "turn_relation_annotations.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert summary["status"] == "ok"
    assert payload["test_turns_processed"] == 2
    assert payload["annotated_turns"] == 1
    assert payload["error_count"] == 1
    assert payload["completed"] is True
    assert len(payload["annotations"]) == 2
    assert payload["annotations"][1]["status"] == "annotation_error"
    assert payload["annotations"][0] == {
        "game_id": 7,
        "turn_number": 1,
        "target_property": "animal",
        "hypothesis": "mammal",
        "status": "ok",
        "relation": "hypothesis_included_in_target",
        "explanation": "ok",
        "error": None,
    }


def test_annotate_results_file_verbose_prints_live_annotations(tmp_path, capsys):
    results_file = tmp_path / "model-live" / "game_results.json"
    results_file.parent.mkdir(parents=True)
    results_file.write_text(
        json.dumps(
            [
                {
                    "game_id": 11,
                    "target_property": "animal",
                    "sampling_hypothesis": "fish",
                    "distance_group": "close",
                    "distance": 2,
                    "turns": [
                        {
                            "turn_number": 1,
                            "action": {"action": "test", "hypothesis": "mammal"},
                            "oracle_response": {"feedback": "one"},
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    class _FakeAnnotator:
        model = "gpt-5-mini"
        provider = "openai"

        def annotate_relation(self, target_property, hypothesis):
            return {
                "status": "ok",
                "relation": "hypothesis_included_in_target",
                "explanation": "subset",
                "raw_response": "{}",
                "error": None,
            }

    annotate_results_file(
        results_file=results_file,
        annotator=_FakeAnnotator(),
        verbose=True,
    )
    captured = capsys.readouterr()

    assert "[model-live] game 11 turn 1" in captured.out
    assert "hypothesis_included_in_target" in captured.out


def test_annotate_results_file_handles_missing_hypothesis_without_local_counter_error(tmp_path):
    results_file = tmp_path / "model-missing" / "game_results.json"
    results_file.parent.mkdir(parents=True)
    results_file.write_text(
        json.dumps(
            [
                {
                    "game_id": 15,
                    "target_property": "animal",
                    "sampling_hypothesis": "fish",
                    "distance_group": "close",
                    "distance": 2,
                    "turns": [
                        {
                            "turn_number": 1,
                            "action": {"action": "test", "hypothesis": ""},
                            "oracle_response": {"feedback": "one"},
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    class _FakeAnnotator:
        model = "gpt-5-mini"
        provider = "openai"

        def annotate_relation(self, target_property, hypothesis):
            raise AssertionError("annotate_relation should not be called for empty hypothesis")

    summary = annotate_results_file(results_file=results_file, annotator=_FakeAnnotator())
    payload = json.loads((results_file.parent / "turn_relation_annotations.json").read_text(encoding="utf-8"))

    assert summary["status"] == "ok"
    assert payload["error_count"] == 1
    assert payload["annotations"][0]["error"] == "Missing or empty test-turn hypothesis."


def test_annotate_results_file_progress_uses_tqdm(tmp_path, monkeypatch):
    results_file = tmp_path / "model-progress" / "game_results.json"
    results_file.parent.mkdir(parents=True)
    results_file.write_text(
        json.dumps(
            [
                {
                    "game_id": 41,
                    "target_property": "animal",
                    "sampling_hypothesis": "fish",
                    "distance_group": "close",
                    "distance": 2,
                    "turns": [
                        {
                            "turn_number": 1,
                            "action": {"action": "test", "hypothesis": "mammal"},
                            "oracle_response": {"feedback": "one"},
                        }
                    ],
                },
                {
                    "game_id": 42,
                    "target_property": "animal",
                    "sampling_hypothesis": "bird",
                    "distance_group": "close",
                    "distance": 2,
                    "turns": [
                        {
                            "turn_number": 1,
                            "action": {"action": "test", "hypothesis": "sparrow"},
                            "oracle_response": {"feedback": "two"},
                        }
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

    class _FakeAnnotator:
        model = "gpt-5-mini"
        provider = "openai"

        def annotate_relation(self, target_property, hypothesis):
            return {
                "status": "ok",
                "relation": "hypothesis_included_in_target",
                "explanation": "ok",
                "raw_response": "{}",
                "error": None,
            }

    events = []

    class _FakeTqdm:
        def __init__(self, **kwargs):
            events.append(("init", kwargs))

        def update(self, value):
            events.append(("update", value))

        def close(self):
            events.append(("close", None))

    monkeypatch.setattr("src.annotation.annotate_relations.tqdm", lambda **kwargs: _FakeTqdm(**kwargs))

    annotate_results_file(
        results_file=results_file,
        annotator=_FakeAnnotator(),
        progress=True,
    )

    assert events[0][0] == "init"
    assert events[0][1]["desc"] == "model-progress games"
    assert ("update", 1) in events
    assert events[-1] == ("close", None)


def test_annotate_many_skips_existing_without_initializing_client(tmp_path, monkeypatch):
    results_file = tmp_path / "results" / "model-x" / "game_results.json"
    results_file.parent.mkdir(parents=True)
    results_file.write_text("[]", encoding="utf-8")
    (results_file.parent / "turn_relation_annotations.json").write_text(
        json.dumps({"completed": True, "annotations": []}),
        encoding="utf-8",
    )

    class _ExplodingAnnotator:
        def __init__(self, *args, **kwargs):
            raise AssertionError("annotator should not be constructed")

    monkeypatch.setattr("src.annotation.annotate_relations.TurnRelationAnnotator", _ExplodingAnnotator)

    summary = annotate_many(input_path=tmp_path / "results", overwrite=False)

    assert summary["status"] == "ok"
    assert summary["processed_files"] == 0
    assert summary["skipped_files"] == 1
    assert summary["processed_models"][0]["status"] == "skipped_existing"


def test_annotate_results_file_persists_partial_checkpoint_before_failure(tmp_path):
    results_file = tmp_path / "model-checkpoint" / "game_results.json"
    results_file.parent.mkdir(parents=True)
    results_file.write_text(
        json.dumps(
            [
                {
                    "game_id": 21,
                    "target_property": "animal",
                    "sampling_hypothesis": "fish",
                    "distance_group": "deep",
                    "distance": 5,
                    "turns": [
                        {
                            "turn_number": 1,
                            "action": {"action": "test", "hypothesis": "mammal"},
                            "oracle_response": {"feedback": "one"},
                        },
                        {
                            "turn_number": 2,
                            "action": {"action": "test", "hypothesis": "bird"},
                            "oracle_response": {"feedback": "two"},
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    class _ExplodingSecondAnnotator:
        model = "gpt-5-mini"
        provider = "openai"

        def __init__(self):
            self.calls = 0

        def annotate_relation(self, target_property, hypothesis):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("boom")
            return {
                "status": "ok",
                "relation": "hypothesis_included_in_target",
                "explanation": "ok",
                "raw_response": "{}",
                "error": None,
            }

    annotator = _ExplodingSecondAnnotator()

    try:
        annotate_results_file(results_file=results_file, annotator=annotator)
    except RuntimeError as error:
        assert str(error) == "boom"
    else:
        raise AssertionError("Expected checkpoint test to raise")

    payload = json.loads((results_file.parent / "turn_relation_annotations.json").read_text(encoding="utf-8"))
    assert payload["completed"] is False
    assert payload["test_turns_processed"] == 1
    assert payload["annotated_turns"] == 1
    assert len(payload["annotations"]) == 1
    assert payload["annotations"][0]["game_id"] == 21
    assert payload["annotations"][0]["turn_number"] == 1
    assert payload["annotations"][0]["target_property"] == "animal"
    assert payload["annotations"][0]["hypothesis"] == "mammal"


def test_annotate_results_file_resumes_from_partial_checkpoint(tmp_path):
    results_file = tmp_path / "model-resume" / "game_results.json"
    results_file.parent.mkdir(parents=True)
    results_file.write_text(
        json.dumps(
            [
                {
                    "game_id": 31,
                    "target_property": "animal",
                    "sampling_hypothesis": "fish",
                    "distance_group": "close",
                    "distance": 3,
                    "turns": [
                        {
                            "turn_number": 1,
                            "action": {"action": "test", "hypothesis": "mammal"},
                            "oracle_response": {"feedback": "one"},
                        },
                        {
                            "turn_number": 2,
                            "action": {"action": "test", "hypothesis": "bird"},
                            "oracle_response": {"feedback": "two"},
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    partial_payload = {
        "source_results_file": str(results_file),
        "annotation_model": "gpt-5-mini",
        "annotation_provider": "openai",
        "relation_labels": [],
        "games_processed": 1,
        "test_turns_processed": 1,
        "annotated_turns": 1,
        "error_count": 0,
        "completed": False,
        "annotations": [
            {
                "game_id": 31,
                "turn_number": 1,
                "target_property": "animal",
                "hypothesis": "mammal",
                "status": "ok",
                "relation": "hypothesis_included_in_target",
                "explanation": "ok",
                "error": None,
            }
        ],
    }
    (results_file.parent / "turn_relation_annotations.json").write_text(
        json.dumps(partial_payload),
        encoding="utf-8",
    )

    class _OneCallAnnotator:
        model = "gpt-5-mini"
        provider = "openai"

        def __init__(self):
            self.calls = []

        def annotate_relation(self, target_property, hypothesis):
            self.calls.append((target_property, hypothesis))
            return {
                "status": "ok",
                "relation": "hypothesis_included_in_target",
                "explanation": "ok",
                "raw_response": "{}",
                "error": None,
            }

    annotator = _OneCallAnnotator()
    summary = annotate_results_file(results_file=results_file, annotator=annotator)
    payload = json.loads((results_file.parent / "turn_relation_annotations.json").read_text(encoding="utf-8"))

    assert summary["resumed_turns"] == 1
    assert payload["completed"] is True
    assert payload["test_turns_processed"] == 2
    assert len(payload["annotations"]) == 2
    assert annotator.calls == [("animal", "bird")]