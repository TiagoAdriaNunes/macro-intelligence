"""Catalog of economic indicators the agent can fetch.

The LLM picks indicators from this catalog by key, so it never invents series codes.
To add an indicator, add an entry here; the planner prompt picks it up automatically.
"""

from dataclasses import dataclass
from typing import Literal

Source = Literal["bcb", "worldbank"]
Frequency = Literal["daily", "monthly", "annual"]

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
    country: str | None = None  # set when the source only covers one country


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
            country="BRA",
        ),
        Indicator(
            key="br_selic_target",
            name="Selic target rate",
            source="bcb",
            code="432",
            units="% per year",
            frequency="daily",
            description="Brazilian policy interest rate target set by Copom.",
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
        ),
        Indicator(
            key="wb_inflation",
            name="Consumer price inflation",
            source="worldbank",
            code="FP.CPI.TOTL.ZG",
            units="% per year",
            frequency="annual",
            description="Annual consumer price inflation for any country.",
        ),
        Indicator(
            key="wb_unemployment",
            name="Unemployment rate",
            source="worldbank",
            code="SL.UEM.TOTL.ZS",
            units="% of labor force, ILO modeled estimate",
            frequency="annual",
            description="Annual unemployment rate for any country, comparable across countries.",
        ),
    ]
}


def catalog_text() -> str:
    """Catalog as bullet lines for the planner prompt."""
    lines = []
    for ind in CATALOG.values():
        scope = "Brazil only" if ind.country == "BRA" else "any country"
        lines.append(f"- {ind.key}: {ind.description} [{ind.frequency}, {ind.units}, {scope}]")
    return "\n".join(lines)
