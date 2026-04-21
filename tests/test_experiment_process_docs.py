"""Tests for repository experiment-process documentation writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gs_sim2real.experiments.process_docs import write_repo_experiment_docs


def _sample_process_section() -> dict[str, Any]:
    return {
        "title": "Widget Selection",
        "updatedAt": "2026-04-21T00:00:00Z",
        "problemStatement": "Pick a widget policy without freezing every experiment.",
        "comparisonHeaders": ["Policy", "Tier", "Score"],
        "comparisonRows": [
            ["Stable Widget", "core", "1.00"],
            ["Prototype Widget", "experiment", "0.50"],
        ],
        "fixtureSections": [],
        "highlights": ["Best policy fit: `Stable Widget`"],
        "accepted": ["Stable code uses `select_widget()` as the only production surface."],
        "deferred": ["Keep `prototype_widget()` in experiments."],
        "rules": ["Production callers depend only on the accepted surface."],
        "stableInterfaceIntro": "Widget callers use one stable function.",
        "stableInterfaceCode": "select_widget(request)",
        "experimentContract": ["Policies receive the same request fixture."],
        "comparableInputs": ["Canonical widget requests."],
        "boundary": ["Experimental policies stay out of production imports."],
    }


def test_write_repo_experiment_docs_splits_public_index_from_detail(tmp_path: Path) -> None:
    outputs = write_repo_experiment_docs([_sample_process_section()], docs_dir=tmp_path)

    assert set(outputs) == {"experiments", "experiments_detail", "decisions", "interfaces"}

    experiments_text = Path(outputs["experiments"]).read_text(encoding="utf-8")
    detail_text = Path(outputs["experiments_detail"]).read_text(encoding="utf-8")

    assert "# Experiments" in experiments_text
    assert "## Current Seams" in experiments_text
    assert "Widget Selection" in experiments_text
    assert "[tables](experiments.generated.md#widget-selection)" in experiments_text
    assert "### Current Comparison" not in experiments_text
    assert "| Stable Widget | core | 1.00 |" not in experiments_text

    assert "# Generated Experiment Comparisons" in detail_text
    assert "Use `docs/experiments.md` as the public index" in detail_text
    assert "### Current Comparison" in detail_text
    assert "| Stable Widget | core | 1.00 |" in detail_text


def test_write_repo_experiment_docs_keeps_decision_and_interface_outputs(tmp_path: Path) -> None:
    outputs = write_repo_experiment_docs([_sample_process_section()], docs_dir=tmp_path)

    decisions_text = Path(outputs["decisions"]).read_text(encoding="utf-8")
    interfaces_text = Path(outputs["interfaces"]).read_text(encoding="utf-8")

    assert "## Widget Selection" in decisions_text
    assert "Stable code uses `select_widget()`" in decisions_text
    assert "Keep `prototype_widget()` in experiments." in decisions_text
    assert "select_widget(request)" in interfaces_text
    assert "Experimental policies stay out of production imports." in interfaces_text
