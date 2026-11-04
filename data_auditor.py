"""Inspect candidate Blackjack CSV schemas without loading whole datasets into memory."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "state_player_sum": ("state_player_sum", "player_sum", "player_total", "player_score"),
    "state_dealer_showing": (
        "state_dealer_showing",
        "dealer_showing",
        "dealer_upcard",
        "dealer_card",
    ),
    "state_usable_ace": ("state_usable_ace", "usable_ace", "player_usable_ace"),
    "action": ("action", "player_action", "decision", "move"),
    "next_player_sum": ("next_player_sum", "next_player_total", "next_player_score"),
    "next_dealer_showing": ("next_dealer_showing", "next_dealer_upcard"),
    "next_usable_ace": ("next_usable_ace", "next_player_usable_ace"),
    "reward": ("reward", "step_reward", "game_reward"),
    "terminated": ("terminated", "done", "is_terminal", "terminal"),
}
OUTCOME_FIELDS = {
    "result",
    "outcome",
    "win",
    "won",
    "winner",
    "final_result",
    "game_result",
    "winloss",
}
ACTION_HISTORY_FIELDS = {
    "actions_taken",
    "action_history",
    "player_actions",
    "actions",
}
HIDDEN_CARD_FIELDS = {
    "dealer_hole_card",
    "hole_card",
    "dealer_hidden_card",
    "hidden_dealer_card",
}


@dataclass(frozen=True)
class AuditReport:
    path: str
    status: str
    columns: list[str]
    mapped_fields: dict[str, str]
    missing_fields: list[str]
    outcome_only: bool
    action_history_columns: list[str]
    hidden_card_columns: list[str]
    rows_inspected: int
    row_width_errors: int
    scan_complete: bool
    notes: list[str]


def audit_csv(path: Path, *, row_limit: int | None = 1_000) -> AuditReport:
    """Audit the header and a bounded number of rows from a UTF-8 CSV file."""
    if row_limit is not None and row_limit < 1:
        raise ValueError("row_limit must be at least 1, or None to scan the entire file")

    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        try:
            raw_header = next(reader)
        except StopIteration:
            raise ValueError(f"{path} is empty") from None

        columns = [name.strip() for name in raw_header]
        normalized = {name.casefold(): name for name in columns}
        mapped = {
            field: normalized[alias]
            for field, aliases in FIELD_ALIASES.items()
            for alias in aliases
            if alias in normalized
        }
        missing = [field for field in FIELD_ALIASES if field not in mapped]
        hidden = sorted(name for name in normalized if name in HIDDEN_CARD_FIELDS)
        action_history = sorted(name for name in normalized if name in ACTION_HISTORY_FIELDS)
        has_outcome = bool(OUTCOME_FIELDS.intersection(normalized))
        outcome_only = has_outcome and not action_history and "action" not in mapped
        row_errors = 0
        rows_inspected = 0
        scan_complete = True

        for row in reader:
            if row_limit is not None and rows_inspected >= row_limit:
                scan_complete = False
                break
            rows_inspected += 1
            if len(row) != len(columns):
                row_errors += 1

    if outcome_only:
        status = "outcome_only"
        notes = [
            "Outcome records do not contain decision transitions and must not be used as transition data."
        ]
    elif action_history and missing:
        status = "terminal_with_action_history"
        notes = [
            "Action history is present, but per-decision state transitions are not; "
            "the data is not directly compatible as transition data."
        ]
    elif missing:
        status = "incomplete_schema"
        notes = ["Required transition fields are missing; schema compatibility is not established."]
    else:
        status = "transition_candidate"
        notes = ["Required transition columns are present; validate values and semantics before use."]

    if hidden:
        notes.append(
            "Potential hidden dealer-card columns are present; exclude them from decision-time observations."
        )
    if row_errors:
        notes.append(f"{row_errors} inspected row(s) have a different field count from the header.")
        if status == "transition_candidate":
            status = "malformed_rows"

    return AuditReport(
        path=str(path),
        status=status,
        columns=columns,
        mapped_fields=mapped,
        missing_fields=missing,
        outcome_only=outcome_only,
        action_history_columns=action_history,
        hidden_card_columns=hidden,
        rows_inspected=rows_inspected,
        row_width_errors=row_errors,
        scan_complete=scan_complete,
        notes=notes,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path, help="Candidate CSV dataset paths")
    parser.add_argument(
        "--row-limit",
        type=int,
        default=1_000,
        help="Rows to inspect per file (default: 1000); use 0 to scan the entire file",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    row_limit = None if args.row_limit == 0 else args.row_limit
    reports = []
    failed = False
    for path in args.datasets:
        try:
            reports.append(asdict(audit_csv(path, row_limit=row_limit)))
        except (OSError, UnicodeError, csv.Error, ValueError) as error:
            failed = True
            reports.append({"path": str(path), "status": "error", "error": str(error)})
    print(json.dumps(reports, indent=2, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
