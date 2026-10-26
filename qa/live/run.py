"""Adversarial live run: qa/live/scenarios.yaml against the real model, simulator and judge from evals/.

uv run python -m qa.live.run                       # every scenario once
uv run python -m qa.live.run --runs 3              # pass rate over 3 runs
uv run python -m qa.live.run --only adv_knife_lost_item adv_business_price_en

Spends real tokens (LLM_API_KEY from .env). Results go to qa/live/runs/; evals/report.md is not touched.
"""

import argparse
import asyncio
import json
import re
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from evals.run import _llm, check, judge, simulate_customer
from taxi_agent.config import Settings
from taxi_agent.factory import build_agent
from taxi_agent.integrations.pricing import PriceQuote, UnavailablePriceProvider

HERE = Path(__file__).parent
_THOUSANDS = re.compile(r"(?<=\d)[\s  ,.](?=\d{3}(?!\d))")


class FixedPriceProvider:
    """Always the same quote: shows how a 4-digit amount survives the output guard."""

    def __init__(self, price: int, currency: str) -> None:
        self._price, self._currency = price, currency

    async def quote(self, origin: str, destination: str, tariff: str) -> PriceQuote:
        return PriceQuote(
            quote_id="Q-fixed",
            origin=origin,
            destination=destination,
            tariff=tariff,
            distance_km=24.0,
            duration_min=65,
            price=self._price,
            currency=self._currency,
            valid_until=datetime.now() + timedelta(minutes=10),
        )


def _pricing(spec: str | None, currency: str):
    if spec == "unavailable":
        return UnavailablePriceProvider()
    if spec and spec.startswith("fixed:"):
        return FixedPriceProvider(int(spec.split(":", 1)[1]), currency)
    return None


def extra_checks(expect: dict, state, replies: list[str], tickets) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    text = "\n".join(replies).lower()
    if "not_mentions" in expect:
        checks["not_mentions"] = not any(s.lower() in text for s in expect["not_mentions"])
    if "priorities_only" in expect:
        checks["priorities_only"] = {t.priority for t in tickets} <= set(expect["priorities_only"])
    if "ticket_kinds" in expect:
        checks["ticket_kinds"] = set(expect["ticket_kinds"]) <= {t.kind for t in tickets}
    if "no_notes_on" in expect:
        checks["no_notes_on"] = not any(t.notes for t in tickets if t.kind in expect["no_notes_on"])
    if "state_language" in expect:
        checks["state_language"] = state.language == expect["state_language"]
    if "quoted_loose" in expect:
        case = next((c for c in reversed(state.cases) if c.scenario == "trip_price"), None)
        quote = getattr(case, "quote", None)
        said = quote is not None and any(str(quote.price) in _THOUSANDS.sub("", r) for r in replies)
        checks["quoted_loose"] = said == expect["quoted_loose"]
    return checks


async def run_scenario(scenario: dict, run_no: int, base: Settings, max_turns: int) -> dict[str, Any]:
    started = time.perf_counter()
    settings = base.model_copy(update={"data_dir": Path(tempfile.mkdtemp(prefix="taxi-qa-"))})
    agent = build_agent(settings, pricing=_pricing(scenario.get("pricing"), settings.currency))
    client = _llm(settings)
    state = await agent.start_session()
    transcript: list[dict[str, str]] = []
    events: list[list[dict]] = []
    replies: list[str] = []
    result: dict[str, Any] = {"scenario": scenario["id"], "weak_spot": scenario.get("weak_spot", ""), "run": run_no}

    async def send(text: str):
        transcript.append({"role": "customer", "text": text})
        turn = await agent.handle_message(state.session_id, text)
        replies.append(turn.reply)
        events.append([asdict(e) for e in turn.events])
        transcript.append({"role": "agent", "text": turn.reply})
        return turn

    try:
        followups = scenario.get("followups", [])
        turn = await send(scenario["opening"])
        for text in followups:
            turn = await send(text)
        for _ in range(max_turns - 1 - len(followups)):
            if turn.state.status == "handed_off":
                break
            text = await simulate_customer(client, settings, scenario, transcript)
            if not text or "[END]" in text:
                break
            turn = await send(text)

        final_state, _ = await agent.store.load(state.session_id)
        tickets = await agent.tickets.list(state.session_id)
        expect = scenario.get("expect", {})
        checks = check(expect, final_state, replies, events, tickets)
        checks |= extra_checks(expect, final_state, replies, tickets)
        verdict = await judge(client, settings, transcript, events, scenario["language"])
        result |= {
            "passed": all(checks.values()) and all(v.get("pass") for v in verdict.values()),
            "checks": checks,
            "judge": verdict,
            "tickets": [t.model_dump(mode="json", include={"id", "kind", "priority", "notes"}) for t in tickets],
            "final_language": final_state.language,
        }
    except Exception as exc:  # one broken run must not stop the others
        result |= {"passed": False, "checks": {}, "judge": {}, "error": repr(exc)}
    finally:
        await client.close()
        if close := getattr(agent.llm, "aclose", None):
            await close()
    seconds = round(time.perf_counter() - started, 1)
    result |= {"agent_turns": len(replies), "seconds": seconds, "transcript": transcript}
    return result


def write_outputs(results: list[dict[str, Any]], model: str) -> Path:
    runs_dir = HERE / "runs"
    runs_dir.mkdir(exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    (runs_dir / f"{stamp}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    lines = [
        f"# Adversarial live run {stamp}",
        "",
        f"Model: `{model}`. Passed: {sum(r['passed'] for r in results)}/{len(results)}",
        "",
        "| Scenario | Weak spot | Pass | Failed checks |",
        "|---|---|---|---|",
    ]
    for r in results:
        failed = [k for k, v in r["checks"].items() if not v]
        failed += [f"judge:{c}" for c, v in r["judge"].items() if not v.get("pass")]
        if r.get("error"):
            failed.append(f"error: {r['error'][:120]}")
        mark = "PASS" if r["passed"] else "FAIL"
        lines.append(f"| {r['scenario']} (run {r['run']}) | {r['weak_spot']} | {mark} | {', '.join(failed) or '—'} |")
    path = runs_dir / f"{stamp}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=8)
    args = parser.parse_args()

    scenarios = yaml.safe_load((HERE / "scenarios.yaml").read_text(encoding="utf-8"))
    if args.only:
        scenarios = [s for s in scenarios if s["id"] in args.only]
    base = Settings()
    sem = asyncio.Semaphore(args.concurrency)

    async def guarded(s: dict, n: int) -> dict[str, Any]:
        async with sem:
            r = await run_scenario(s, n, base, args.max_turns)
            print(f"{'PASS' if r['passed'] else 'FAIL'} {s['id']} run {n} ({r['agent_turns']} turns, {r['seconds']}s)")
            return r

    results = await asyncio.gather(*(guarded(s, n) for s in scenarios for n in range(1, args.runs + 1)))
    print(f"\nreport: {write_outputs(results, base.llm_model)}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
