"""Tables by year, built from a dataset summary (analysis.summarize), not by the LLM.

The chat shows them with great_tables; the command line prints them as plain text.
"""

from uuid import uuid4

import pandas as pd
from great_tables import GT

COLUMNS = {
    "year": "Year",
    "date": "Date",
    "value": "Year-end value",
    "change": "Change",
    "pct_change": "Change (%)",
    "mean": "Year average",
}


def yearly_frame(summary: dict) -> pd.DataFrame:
    """One row per year: year-end value (the latest, for the current year), its date, the
    change from the previous year-end and, for monthly or daily data, the year's average."""
    rows = [
        {
            "year": row["year"],
            "date": row["end"]["date"],
            "value": row["end"]["value"],
            "change": row.get("change"),
            "pct_change": row.get("pct_change"),
            "mean": row.get("mean"),
        }
        for row in summary.get("yearly", [])
    ]
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    return frame.dropna(axis="columns", how="all")  # e.g. no % change for rates


def _decimals(summary: dict, column: str) -> int:
    if column == "pct_change" or "%" in summary["units"]:
        return 2
    return 4  # levels such as BRL per USD


def _theme_css(table_id: str) -> str:
    """Bootstrap's theme colors instead of GT's fixed white and greys, so the table follows
    the chat's light or dark mode. It comes after GT's own CSS, so it takes precedence."""
    selectors = f"#{table_id} .gt_table, #{table_id} .gt_table *"
    return (
        f"{selectors} {{ background-color: transparent; color: var(--bs-body-color); "
        "border-color: var(--bs-border-color); }"
    )


def yearly_table_html(summary: dict) -> str:
    """The table for the chat, as HTML."""
    frame = yearly_frame(summary)
    change_units = summary.get("change_units", summary["units"])
    table_id = f"yearly-{uuid4().hex[:8]}"  # unique in the chat, which keeps earlier replies
    table = (
        GT(frame, id=table_id)
        .tab_header(
            title=f"{summary['name']} ({summary['country']})",
            subtitle=f"{summary['units']}; changes in {change_units}",
        )
        .cols_label(cases={c: COLUMNS[c] for c in frame.columns})
    )
    for column in frame.columns.intersection(["value", "mean", "change", "pct_change"]):
        table = table.fmt_number(
            columns=column,
            decimals=_decimals(summary, column),
            force_sign=column in ("change", "pct_change"),
        )
    table = (
        table.sub_missing(missing_text="")
        .tab_source_note(
            f"Source: {summary['source']}, series {summary['code']}. The current year shows "
            "its latest value."
        )
        .tab_options(table_font_size="14px")
        .opt_css(_theme_css(table_id))
    )
    return table.as_raw_html()


def yearly_table_text(summary: dict) -> str:
    """The table for the command line, as aligned plain text."""
    frame = yearly_frame(summary)

    def cell(column: str, value: object) -> str:
        if not isinstance(value, float):
            return str(value)  # year, date
        if pd.isna(value):
            return ""
        sign = "+" if column in ("change", "pct_change") else ""
        return f"{value:{sign}.{_decimals(summary, column)}f}"

    header = [COLUMNS[c] for c in frame.columns]
    body = [
        [cell(c, v) for c, v in zip(frame.columns, row, strict=True)]
        for row in frame.itertuples(index=False)
    ]
    widths = [max(len(r[i]) for r in [header, *body]) for i in range(len(header))]
    lines = [
        "  ".join(text.rjust(width) for text, width in zip(r, widths, strict=True))
        for r in [header, *body]
    ]
    title = f"{summary['name']} ({summary['country']}), {summary['units']}"
    return "\n".join([title, *lines])
