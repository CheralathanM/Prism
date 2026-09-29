"""FDB-v3 adapter: the 12 benchmark tool specs and a backend bound to the official mocks.

Tool names and parameters mirror the official templates (``v3/lk_agent_tool.py``) exactly,
because the evaluator compares function names and argument names. Descriptions are our own.
Execution goes through the unmodified ``v3/mock_apis.py``; each executed call is appended
to the harness telemetry file in the template's exact format.

State-changing classification (ours; drives the commit gate and idempotency guard):
book_flight, update_identity_doc, modify_autopay, update_search_filter, add_to_cart.
"""

from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any

from fdagent.core.model import ToolParam, ToolSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FDB_V3_DIR = REPO_ROOT / "third_party" / "Full-Duplex-Bench" / "v3"
# The harness reads this exact path (run_tool_benchmark.py, "Step 6").
DEFAULT_TOOL_LOG = "/tmp/agent_tool_calls.log"


def _s(name: str, desc: str, required: bool = True) -> ToolParam:
    return ToolParam(name, "string", required, desc)


def _n(name: str, desc: str, required: bool = True, integer: bool = False) -> ToolParam:
    return ToolParam(name, "integer" if integer else "number", required, desc)


FDB_TOOL_SPECS: tuple[ToolSpec, ...] = (
    # Travel & identity
    ToolSpec("search_flights", "Search available flights to a destination on a date.",
             (_s("destination", "City or airport, e.g. 'London' or 'LHR'"),
              _s("date", "Travel date as spoken or ISO, e.g. 'August 20' or '2026-08-20'"))),
    ToolSpec("book_flight", "Book a flight ticket for a passenger.",
             (_s("passenger_name", "Full name of the passenger"),), state_changing=True),
    ToolSpec("update_identity_doc", "Update the user's identity document on file (simulated).",
             (_s("doc_type", "Document type, e.g. 'passport' or 'id_card'"),
              _s("doc_number", "Document number")), state_changing=True),
    # Finance & billing
    ToolSpec("get_card_benefits", "Look up the benefits of a credit card type.",
             (_s("card_type", "Card type, e.g. 'platinum' or 'gold'"),)),
    ToolSpec("get_exchange_rate", "Convert an amount between two currencies using the live rate.",
             (_n("amount", "Amount to convert"),
              _s("from_currency", "3-letter code, e.g. 'USD'"),
              _s("to_currency", "3-letter code, e.g. 'EUR'"))),
    ToolSpec("modify_autopay", "Change which account pays a bill automatically.",
             (_s("bill_type", "Bill type, e.g. 'credit_card' or 'utilities'"),
              _s("source_account", "Paying account, e.g. 'checking'")), state_changing=True),
    # Housing & location
    ToolSpec("search_apartments", "Search rental apartments.",
             (_s("city", "City"), _n("bedrooms", "Number of bedrooms", integer=True),
              _n("max_price", "Maximum monthly rent"))),
    ToolSpec("calculate_commute", "Calculate commute time between two places.",
             (_s("origin_address", "Starting location"), _s("destination_address", "Destination"),
              _s("mode", "driving, transit, walking or cycling (default driving)", required=False))),
    ToolSpec("update_search_filter", "Update one of the user's saved search filters.",
             (_s("filter_name", "Filter key to modify"),
              _s("value", "New value for the filter")), state_changing=True),
    # E-commerce
    ToolSpec("track_order", "Get the shipping status of an order.",
             (_s("order_id", "Order identifier"),)),
    ToolSpec("search_products", "Search the product catalog.",
             (_s("query", "What to search for"), _n("max_price", "Optional budget", required=False))),
    ToolSpec("add_to_cart", "Add a product to the shopping cart.",
             (_s("product_id", "Product id, usually from search results"),
              _n("quantity", "How many (default 1)", required=False, integer=True)), state_changing=True),
)

# Defaults the templates always pass (and log) when the model omits them.
_TEMPLATE_DEFAULTS: dict[str, dict[str, Any]] = {
    "calculate_commute": {"mode": "driving"},
    "search_products": {"max_price": None},
    "add_to_cart": {"quantity": 1},
}


def load_mock_registry_class(fdb_v3_dir: str | Path | None = None):
    """Import ``MockAPIRegistry`` from the pinned, unmodified FDB-v3 checkout."""
    d = Path(fdb_v3_dir or os.getenv("FDB_V3_DIR") or DEFAULT_FDB_V3_DIR)
    if not (d / "mock_apis.py").exists():
        raise FileNotFoundError(f"FDB-v3 mock_apis.py not found in {d}; run scripts/fetch_fdb.sh")
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    from mock_apis import MockAPIRegistry  # type: ignore[import-not-found]

    return MockAPIRegistry


class FdbMockBackend:
    """Synchronous backend (the executor runs it in a worker thread).

    One instance per session/room: the registry's latency injector keeps per-API call
    counters, so sharing it across scenarios would leak state (INVARIANT 12).
    """

    _log_lock = threading.Lock()

    def __init__(
        self,
        room_name: str,
        latency_profile: str = "instant",
        tool_log_path: str | None = None,
        fdb_v3_dir: str | Path | None = None,
    ) -> None:
        self.room_name = room_name
        self.tool_log_path = tool_log_path or os.getenv("FDB_TOOL_LOG", DEFAULT_TOOL_LOG)
        self.registry = load_mock_registry_class(fdb_v3_dir)(latency_profile=latency_profile)

    @staticmethod
    def seed(seed: int) -> None:
        """Seed the global RNG used by the harness latency injector (jitter only; tool
        outputs are deterministic regardless)."""
        random.seed(seed)

    def call(self, tool: str, args: dict[str, Any], idempotency_key: str) -> Any:
        full = {**_TEMPLATE_DEFAULTS.get(tool, {}), **args}
        t0 = time.time()
        result = self.registry.call(tool, **full)
        t1 = time.time()
        self._log(tool, full, t0, t1)
        return result

    def _log(self, tool: str, args: dict[str, Any], t0: float, t1: float) -> None:
        line = json.dumps(
            {"room": self.room_name,
             "call": {"function": tool, "args": args, "timestamp_start": t0, "timestamp_end": t1}}
        )
        path = Path(self.tool_log_path)
        with self._log_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
