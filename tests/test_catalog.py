"""Data contracts and ontology (contracts.py, ontology.py): valid, consistent, up to date.

tests/schemas/odcs-json-schema-v3.1.0.json is the official ODCS JSON Schema (Apache-2.0), from
https://github.com/bitol-io/open-data-contract-standard/tree/main/schema
"""

import json
from pathlib import Path

import pytest
from jsonschema.validators import validator_for

from macro_intelligence.contracts import CATALOG_DIR, contract, rendered
from macro_intelligence.indicators import CATALOG
from macro_intelligence.ontology import CONCEPTS

SCHEMA = json.loads(
    (Path(__file__).parent / "schemas" / "odcs-json-schema-v3.1.0.json").read_text("utf-8")
)


@pytest.mark.parametrize("key", list(CATALOG))
def test_contract_is_valid_odcs(key):
    validator = validator_for(SCHEMA)(SCHEMA)
    errors = [error.message for error in validator.iter_errors(contract(CATALOG[key]))]
    assert errors == []


def test_catalog_files_are_up_to_date():
    files = rendered()
    on_disk = {
        p.relative_to(CATALOG_DIR).as_posix()
        for p in CATALOG_DIR.rglob("*")
        if p.suffix in (".yaml", ".md")
    }
    assert on_disk == set(files), "run: uv run python -m macro_intelligence.contracts"
    for relative, text in files.items():
        assert (CATALOG_DIR / relative).read_text("utf-8") == text, (
            f"{relative} is out of date; run: uv run python -m macro_intelligence.contracts"
        )


def test_every_indicator_measures_a_known_concept():
    for ind in CATALOG.values():
        assert ind.concept in CONCEPTS, ind.key
        if ind.derived_from:  # derived from an indicator of the same concept
            assert CATALOG[ind.derived_from].concept == ind.concept, ind.key


def test_every_concept_is_measured():
    measured = {ind.concept for ind in CATALOG.values()}
    assert set(CONCEPTS) == measured
