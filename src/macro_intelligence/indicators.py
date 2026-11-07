"""Catalog of economic indicators the agent can fetch.

The LLM picks indicators from this catalog by key, so it never invents series codes.
To add an indicator, add an entry here; the planner prompt picks it up automatically.
"""

from dataclasses import dataclass
from typing import Literal

Source = Literal["bcb", "worldbank"]

SOURCE_NAMES: dict[str, str] = {
    "bcb": "Banco Central do Brasil (SGS)",
    "worldbank": "World Bank (World Development Indicators)",
}


@dataclass(frozen=True)
class Indicator:
    key: str
    name: str
    source: Source
    code: str
    units: str
    frequency: str  # as published by the source
    description: str
    country: str | None = None  # set when the source only covers one country


CATALOG: dict[str, Indicator] = {
    i.key: i
    for i in [
        Indicator(
            "br_ipca_monthly",
            "IPCA inflation (monthly change)",
            "bcb",
            "433",
            "% change from previous month",
            "monthly",
            "Brazilian consumer price inflation, month-over-month.",
            "BRA",
        ),
        Indicator(
            "br_ipca_12m",
            "IPCA inflation (12-month)",
            "bcb",
            "13522",
            "% change over 12 months",
            "monthly",
            "Brazilian consumer price inflation accumulated over 12 months.",
            "BRA",
        ),
        Indicator(
            "br_selic_target",
            "Selic target rate",
            "bcb",
            "432",
            "% per year",
            "daily",
            "Brazilian policy interest rate target set by Copom.",
            "BRA",
        ),
        Indicator(
            "br_usd_brl",
            "USD/BRL exchange rate",
            "bcb",
            "1",
            "BRL per USD",
            "daily",
            "Commercial US dollar exchange rate (selling).",
            "BRA",
        ),
        Indicator(
            "br_unemployment",
            "Unemployment rate (PNAD Continua)",
            "bcb",
            "24369",
            "% of labor force",
            "monthly",
            "Brazilian unemployment rate, rolling quarter, from IBGE PNAD Continua.",
            "BRA",
        ),
        Indicator(
            "wb_gdp_growth",
            "GDP growth",
            "worldbank",
            "NY.GDP.MKTP.KD.ZG",
            "% per year, constant prices",
            "annual",
            "Real GDP growth for any country.",
        ),
        Indicator(
            "wb_inflation",
            "Consumer price inflation",
            "worldbank",
            "FP.CPI.TOTL.ZG",
            "% per year",
            "annual",
            "Annual consumer price inflation for any country.",
        ),
        Indicator(
            "wb_unemployment",
            "Unemployment rate",
            "worldbank",
            "SL.UEM.TOTL.ZS",
            "% of labor force, ILO modeled estimate",
            "annual",
            "Annual unemployment rate for any country, comparable across countries.",
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
