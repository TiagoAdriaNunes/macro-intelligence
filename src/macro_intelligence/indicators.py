"""Catalog of economic indicators the agent can fetch.

The LLM picks indicators from this catalog by key, so it never invents series codes.
To add an indicator, add an entry here; the planner prompt picks it up automatically. Then
regenerate the data contracts (see contracts.py).
"""

from dataclasses import dataclass
from typing import Literal

from .ontology import CONCEPTS

Source = Literal["bcb", "worldbank"]
Frequency = Literal["daily", "monthly", "annual"]

# Frequency of a daily series once sources.py keeps only the last value of each month
MONTH_ENDS = "monthly (last daily value of each month)"

SOURCE_NAMES: dict[Source, str] = {
    "bcb": "Banco Central do Brasil (SGS)",
    "worldbank": "World Bank (World Development Indicators)",
}


@dataclass(frozen=True, kw_only=True)
class Indicator:
    key: str
    name: str
    source: Source
    code: str
    units: str
    frequency: Frequency  # as published by the source
    description: str
    concept: str  # the ontology concept it measures (ontology.CONCEPTS)
    publication_lag_days: int  # about how long after its period a value is published
    country: str | None = None  # set when the source only covers one country
    derived_from: str | None = None  # key of the indicator it's calculated from


CATALOG: dict[str, Indicator] = {
    i.key: i
    for i in [
        Indicator(
            key="br_ipca_monthly",
            name="IPCA inflation (monthly change)",
            source="bcb",
            code="433",
            units="% change from previous month",
            frequency="monthly",
            description="Brazilian consumer price inflation, month-over-month.",
            concept="inflation",
            publication_lag_days=15,  # IBGE publishes around the 10th of the next month
            country="BRA",
        ),
        Indicator(
            key="br_ipca_12m",
            name="IPCA inflation (12-month)",
            source="bcb",
            code="13522",
            units="% change over 12 months",
            frequency="monthly",
            description="Brazilian consumer price inflation accumulated over 12 months.",
            concept="inflation",
            publication_lag_days=15,
            country="BRA",
            derived_from="br_ipca_monthly",
        ),
        Indicator(
            key="br_selic_target",
            name="Selic target rate",
            source="bcb",
            code="432",
            units="% per year",
            frequency="daily",
            description="Brazilian policy interest rate target set by Copom.",
            concept="policy_rate",
            publication_lag_days=1,
            country="BRA",
        ),
        Indicator(
            key="br_usd_brl",
            name="USD/BRL exchange rate",
            source="bcb",
            code="1",
            units="BRL per USD",
            frequency="daily",
            description="Commercial US dollar exchange rate (selling).",
            concept="exchange_rate",
            publication_lag_days=1,
            country="BRA",
        ),
        Indicator(
            key="br_unemployment",
            name="Unemployment rate (PNAD Continua)",
            source="bcb",
            code="24369",
            units="% of labor force",
            frequency="monthly",
            description="Brazilian unemployment rate, rolling quarter, from IBGE PNAD Continua.",
            concept="unemployment",
            publication_lag_days=35,  # IBGE publishes about a month after the quarter
            country="BRA",
        ),
        Indicator(
            key="wb_gdp_growth",
            name="GDP growth",
            source="worldbank",
            code="NY.GDP.MKTP.KD.ZG",
            units="% per year, constant prices",
            frequency="annual",
            description="Real GDP growth for any country.",
            concept="economic_growth",
            publication_lag_days=210,  # a year's figures come out during the next year
        ),
        Indicator(
            key="wb_inflation",
            name="Consumer price inflation",
            source="worldbank",
            code="FP.CPI.TOTL.ZG",
            units="% per year",
            frequency="annual",
            description="Annual consumer price inflation for any country.",
            concept="inflation",
            publication_lag_days=210,
        ),
        Indicator(
            key="wb_unemployment",
            name="Unemployment rate",
            source="worldbank",
            code="SL.UEM.TOTL.ZS",
            units="% of labor force, ILO modeled estimate",
            frequency="annual",
            description="Annual unemployment rate for any country, comparable across countries.",
            concept="unemployment",
            publication_lag_days=210,
        ),
    ]
}


def catalog_markdown() -> str:
    """Catalog for people, by the countries it covers."""
    brazil = [f"- {i.name}, {i.frequency}" for i in CATALOG.values() if i.country == "BRA"]
    world = [f"- {i.name}, {i.frequency}" for i in CATALOG.values() if i.country is None]
    return (
        "**Brazil**\n\n" + "\n".join(brazil) + "\n\n"
        "**Any country** (compare several at once)\n\n" + "\n".join(world)
    )


def catalog_text() -> str:
    """Catalog as bullet lines for the planner prompt, with the concept each one measures."""
    lines = []
    for ind in CATALOG.values():
        scope = "Brazil only" if ind.country == "BRA" else "any country"
        concept = CONCEPTS[ind.concept].name
        lines.append(
            f"- {ind.key}: {ind.description} [{concept}; {ind.frequency}, {ind.units}, {scope}]"
        )
    return "\n".join(lines)
