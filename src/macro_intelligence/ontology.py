"""A light ontology: the economic concepts the indicators measure.

    Concept <--measures-- Indicator --derived_from--> Indicator
                              |
                              +--published_by--> Source (indicators.SOURCE_NAMES)

Each Indicator in indicators.py names the concept it measures, and optionally the indicator it
is derived from. Indicators of the same concept are related but not always comparable (e.g.
monthly vs. annual inflation, or two unemployment methodologies).
"""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class Concept:
    key: str
    name: str
    definition: str


CONCEPTS: dict[str, Concept] = {
    c.key: c
    for c in [
        Concept(
            key="inflation",
            name="Inflation",
            definition="The rise in consumer prices over time.",
        ),
        Concept(
            key="policy_rate",
            name="Policy interest rate",
            definition="The interest rate a central bank sets to steer inflation and activity.",
        ),
        Concept(
            key="exchange_rate",
            name="Exchange rate",
            definition="The price of one currency in terms of another.",
        ),
        Concept(
            key="unemployment",
            name="Unemployment",
            definition="The share of the labor force without a job and looking for one.",
        ),
        Concept(
            key="economic_growth",
            name="Economic growth",
            definition="The change in the volume of goods and services an economy produces "
            "(real GDP).",
        ),
    ]
}
