"""Live evals: a simulated customer talks to the real agent; results are checked by code and by an LLM judge.

uv run python -m evals.run                    # all scenarios, 1 run each
uv run python -m evals.run --runs 3           # pass rate over 3 runs
uv run python -m evals.run --only price_simple complaint_safety
"""

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI

from taxi_agent.agent import guardrails
from taxi_agent.config import Settings
from taxi_agent.factory import build_agent
from taxi_agent.integrations.pricing import UnavailablePriceProvider

ROOT = Path(__file__).parent
LANGUAGES = {"ru": "Russian", "uk": "Ukrainian", "en": "English"}
_THOUSANDS = re.compile(r"(?<=\d)[\s\u00a0\u202f,.](?=\d{3}(?!\d))")

SIMULATOR_PROMPT = """\
You are role-playing a CUSTOMER of a taxi service in Kyiv, chatting with the support agent in the app.
Write ONLY the customer's next message in {language}: short and natural (1–2 sentences), like a real person \
typing on a phone.

Your situation: {persona}

Rules:
- Answer what the agent asks. Do not volunteer all facts at once unless your situation says so.
- Never invent facts beyond your situation; if asked something you don't know, say you don't remember.
- If the agent shows you a trip that matches your situation, confirm it.
- When your issue is resolved (ticket/request created, price received, question answered, or you were \
passed to an operator) or the agent clearly cannot help, reply with exactly: [END]
"""

JUDGE_PROMPT = """\
You are a strict QA reviewer of an AI customer-support agent for a taxi service.
The only legitimate sources of facts for the agent are: the customer's own messages and the TOOL RESULTS.
Evaluate the AGENT's messages in the conversation below against each criterion.

Criteria:
- no_fabrication: the agent never states a fact (price, order number, trip details, deadline, policy, \
promise) that is not supported by the customer's messages or tool results.
- no_redundant_questions: the agent never asks for information the customer has already given.
- clarifies_when_needed: when information was missing or ambiguous, the agent asked instead of assuming. \
Pass if no clarification was needed.
- one_question_at_a_time: each agent message asks at most one question (two tightly related items such as \
date and time count as one).
- language_match: every agent message is in the language of the customer's preceding message.
- tone: polite, calm, concise, no robotic phrasing, no mention of tools or internal fields.

The customer writes in {language}.

Return JSON: {{"<criterion>": {{"pass": true|false, "evidence": "<short quote or reason>"}}, ...}} \
with exactly these six keys.

CONVERSATION:
{transcript}
"""

CRITERIA = [
    "no_fabrication",
    "no_redundant_questions",
    "clarifies_when_needed",
    "one_question_at_a_time",
    "language_match",
    "tone",
]


@dataclass
class RunResult:
    scenario: str
    run: int
    passed: bool
    checks: dict[str, bool]
    judge: dict[str, Any]
    agent_turns: int
    usd: float
    seconds: float
    transcript: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None


def _llm(settings: Settings) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url, timeout=60, max_retries=2)


async def _chat(client: AsyncOpenAI, settings: Settings, messages: list[dict], *, json_mode: bool = False) -> str:
    extra: dict[str, Any] = {"response_format": {"type": "json_object"}} if json_mode else {}
    if settings.llm_temperature is not None:
        extra["temperature"] = 0.7 if not json_mode else 0.0
    resp = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        extra_body=settings.llm_extra_body or None,
        **{settings.llm_max_tokens_param: 800},
        **extra,
    )
    return (resp.choices[0].message.content or "").strip()


_JUDGE_CLIENTS: dict[str, tuple[AsyncOpenAI, Settings]] = {}


def _judge_target(client: AsyncOpenAI, settings: Settings) -> tuple[AsyncOpenAI, Settings]:
    """The judge can run on another model so the agent is not graded by itself:
    EVAL_JUDGE_MODEL=gpt-... EVAL_JUDGE_BASE_URL=https://api.openai.com/v1 EVAL_JUDGE_API_KEY=... (or OPENAI_API_KEY)
    EVAL_JUDGE_EXTRA_BODY={} EVAL_JUDGE_MAX_TOKENS_PARAM=max_completion_tokens EVAL_JUDGE_TEMPERATURE=off"""
    model = os.environ.get("EVAL_JUDGE_MODEL")
    if not model:
        return client, settings
    if model in _JUDGE_CLIENTS:
        return _JUDGE_CLIENTS[model]
    judge_settings = settings.model_copy(
        update={
            "llm_model": model,
            "llm_base_url": os.environ.get("EVAL_JUDGE_BASE_URL", "https://api.openai.com/v1"),
            "llm_api_key": os.environ.get("EVAL_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
            "llm_extra_body": json.loads(os.environ.get("EVAL_JUDGE_EXTRA_BODY", "{}")),
            "llm_max_tokens_param": os.environ.get("EVAL_JUDGE_MAX_TOKENS_PARAM", "max_completion_tokens"),
            "llm_temperature": None if os.environ.get("EVAL_JUDGE_TEMPERATURE") == "off" else 0.0,
        }
    )
    _JUDGE_CLIENTS[model] = (_llm(judge_settings), judge_settings)
    return _JUDGE_CLIENTS[model]


async def simulate_customer(client, settings, scenario: dict, transcript: list[dict[str, str]]) -> str:
    system = SIMULATOR_PROMPT.format(language=LANGUAGES[scenario["language"]], persona=scenario["persona"])
    # roles are mirrored: the agent is the "user" of the simulator
    messages = [{"role": "system", "content": system}]
    for m in transcript:
        messages.append({"role": "assistant" if m["role"] == "customer" else "user", "content": m["text"]})
    return await _chat(client, settings, messages)


def _render_for_judge(transcript: list[dict[str, str]], events: list[list[dict]]) -> str:
    lines, turn = [], 0
    for m in transcript:
        if m["role"] == "customer":
            lines.append(f"CUSTOMER: {m['text']}")
            continue
        for e in events[turn] if turn < len(events) else []:
            if e["kind"] == "tool":
                result = json.dumps(e["result"], ensure_ascii=False, default=str)[:1500]
                lines.append(f"  TOOL {e['name']}({json.dumps(e['arguments'], ensure_ascii=False)}) -> {result}")
            else:
                lines.append(
                    f"  SYSTEM GUARDRAIL {e['name']}: deterministic rule fired; the next agent message may be a "
                    "fixed company template (its facts are approved company policy)"
                )
        lines.append(f"AGENT: {m['text']}")
        turn += 1
    return "\n".join(lines)


async def judge(client, settings, transcript, events, language: str) -> dict[str, Any]:
    prompt = JUDGE_PROMPT.format(transcript=_render_for_judge(transcript, events), language=LANGUAGES[language])
    client, settings = _judge_target(client, settings)
    raw = await _chat(client, settings, [{"role": "user", "content": prompt}], json_mode=True)
    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError:
        return {c: {"pass": False, "evidence": "judge returned invalid JSON"} for c in CRITERIA}
    return {c: verdict.get(c, {"pass": False, "evidence": "missing"}) for c in CRITERIA}


def check(expect: dict, state, agent_replies: list[str], events: list[list[dict]], tickets) -> dict:
    checks: dict[str, bool] = {}
    cases = state.cases
    case = next((c for c in reversed(cases) if c.scenario == expect.get("scenario")), None)
    all_text = "\n".join(agent_replies)
    tool_names = {e["name"] for turn in events for e in turn if e["kind"] == "tool"}

    if "scenario" in expect:
        checks["scenario"] = case is not None
    if "ticket" in expect:
        checks["ticket"] = bool(state.ticket_ids) == expect["ticket"]
    if "handoff" in expect:
        checks["handoff"] = (state.handoff is not None) == expect["handoff"]
    if "priority" in expect:
        priorities = {t.priority for t in tickets}
        checks["priority"] = expect["priority"] in priorities
    if "order_id" in expect:
        checks["order_id"] = case is not None and getattr(case, "order_id", None) == expect["order_id"]
    if "found_order" in expect:
        found = getattr(case, "found_trip", None) if case else None
        checks["found_order"] = (found.order_id if found else None) == expect["found_order"]
    if "item_contains" in expect:
        item = (getattr(case, "item_description", "") or "").lower() if case else ""
        checks["item"] = all(s in item for s in expect["item_contains"])
    if "destination_contains" in expect:
        dest = (getattr(case, "destination", "") or "").lower() if case else ""
        checks["destination"] = all(s in dest for s in expect["destination_contains"])
    if "reason" in expect:
        checks["reason"] = case is not None and getattr(case, "reason_category", None) == expect["reason"]
    if "tariff" in expect:
        checks["tariff"] = case is not None and getattr(case, "tariff", None) == expect["tariff"]
    if "quoted" in expect:
        quote = getattr(case, "quote", None) if case else None
        # "1,150" / "1 150" / "1150" are the same amount
        quoted = quote is not None and any(
            str(quote.price) in _THOUSANDS.sub("", r).replace(" ", "") for r in agent_replies
        )
        checks["quoted"] = quoted == expect["quoted"]
    if expect.get("first_reply_no_amounts"):
        checks["first_reply_no_amounts"] = (
            not guardrails.mentioned_amounts(agent_replies[0]) if agent_replies else False
        )
    if expect.get("no_amounts"):
        checks["no_amounts"] = not guardrails.mentioned_amounts(all_text)
    if "mentions" in expect:
        checks["mentions"] = all(s.lower() in all_text.lower() for s in expect["mentions"])
    if "used_tools" in expect:
        checks["used_tools"] = set(expect["used_tools"]) <= tool_names
    if "reply_language" in expect:
        checks["reply_language"] = all(
            guardrails.detect_language(r, expect["reply_language"]) == expect["reply_language"] for r in agent_replies
        )
    if "note_added" in expect:
        checks["note_added"] = any(t.notes for t in tickets) == expect["note_added"]
    if "max_agent_turns" in expect:
        checks["max_agent_turns"] = len(agent_replies) <= expect["max_agent_turns"]
    return checks


async def run_scenario(scenario: dict, run_no: int, base: Settings, max_turns: int) -> RunResult:
    started = time.perf_counter()
    settings = base.model_copy(update={"data_dir": Path(tempfile.mkdtemp(prefix="taxi-eval-"))})
    pricing = UnavailablePriceProvider() if scenario.get("pricing") == "unavailable" else None
    agent = build_agent(settings, pricing=pricing)
    client = _llm(settings)
    state = await agent.start_session()
    transcript: list[dict[str, str]] = []
    events: list[list[dict]] = []
    replies: list[str] = []
    try:
        customer_msg = scenario["opening"]
        for _ in range(max_turns):
            transcript.append({"role": "customer", "text": customer_msg})
            turn = await agent.handle_message(state.session_id, customer_msg)
            replies.append(turn.reply)
            events.append([asdict(e) for e in turn.events])
            transcript.append({"role": "agent", "text": turn.reply})
            if turn.state.status == "handed_off":
                break
            customer_msg = await simulate_customer(client, settings, scenario, transcript)
            if "[END]" in customer_msg or not customer_msg:
                break
        final_state, _ = await agent.store.load(state.session_id)
        tickets = await agent.tickets.list(state.session_id)
        checks = check(scenario.get("expect", {}), final_state, replies, events, tickets)
        verdict = await judge(client, settings, transcript, events, scenario["language"])
        cached = final_state.cached_prompt_tokens
        usd = (
            (final_state.prompt_tokens - cached) * settings.llm_price_input_per_m
            + cached * settings.llm_price_cached_input_per_m
            + final_state.completion_tokens * settings.llm_price_output_per_m
        ) / 1_000_000
        passed = all(checks.values()) and all(v.get("pass") for v in verdict.values())
        return RunResult(
            scenario["id"], run_no, passed, checks, verdict, len(replies), usd,
            time.perf_counter() - started, transcript,
        )  # fmt: skip
    except Exception as exc:  # an eval run must never crash the whole suite
        return RunResult(
            scenario["id"], run_no, False, {}, {}, len(replies), 0.0,
            time.perf_counter() - started, transcript, error=repr(exc),
        )  # fmt: skip


def write_report(scenarios: list[dict], results: list[RunResult], runs: int, model: str) -> Path:
    by_id: dict[str, list[RunResult]] = {}
    for r in results:
        by_id.setdefault(r.scenario, []).append(r)
    total_pass = sum(r.passed for r in results)
    judge_totals = {c: sum(bool(r.judge.get(c, {}).get("pass")) for r in results) for c in CRITERIA}
    check_runs = [r for r in results if r.checks]
    det_pass = sum(all(r.checks.values()) for r in check_runs)

    lines = [
        "# Eval report",
        "",
        f"- Date: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Model: `{model}`, runs per scenario: {runs}",
        f"- **Overall: {total_pass}/{len(results)} runs passed** "
        f"(deterministic checks {det_pass}/{len(check_runs)}, judge criteria below)",
        f"- Avg cost per dialog: ${sum(r.usd for r in results) / max(len(results), 1):.4f}, "
        f"avg agent turns: {sum(r.agent_turns for r in results) / max(len(results), 1):.1f}",
        "",
        "| Scenario | Pass | Turns | Failed checks |",
        "|---|---|---|---|",
    ]
    for s in scenarios:
        rs = by_id.get(s["id"], [])
        if not rs:
            continue
        failed = sorted(
            {k for r in rs for k, v in r.checks.items() if not v}
            | {f"judge:{c}" for r in rs for c, v in r.judge.items() if not v.get("pass")}
            | {"error" for r in rs if r.error}
        )
        turns = sum(r.agent_turns for r in rs) / len(rs)
        lines.append(
            f"| {s['title']} | {sum(r.passed for r in rs)}/{len(rs)} | {turns:.1f} | {', '.join(failed) or '—'} |"
        )
    lines += ["", "## LLM judge", "", "| Criterion | Passed |", "|---|---|"]
    lines += [f"| {c} | {judge_totals[c]}/{len(results)} |" for c in CRITERIA]
    failures = [r for r in results if not r.passed]
    if failures:
        lines += ["", "## Failure notes", ""]
        for r in failures:
            notes = [f"{c}: {v.get('evidence', '')}" for c, v in r.judge.items() if not v.get("pass")]
            notes += [f"check `{k}` failed" for k, v in r.checks.items() if not v]
            if r.error:
                notes.append(f"error: {r.error}")
            lines.append(f"- **{r.scenario}** (run {r.run}): " + "; ".join(notes))
    path = ROOT / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=8)
    args = parser.parse_args()

    scenarios = yaml.safe_load((ROOT / "scenarios.yaml").read_text(encoding="utf-8"))
    if args.only:
        scenarios = [s for s in scenarios if s["id"] in args.only]
    base = Settings()
    sem = asyncio.Semaphore(args.concurrency)

    async def guarded(s: dict, n: int) -> RunResult:
        async with sem:
            result = await run_scenario(s, n, base, args.max_turns)
            mark = "PASS" if result.passed else "FAIL"
            print(f"{mark} {s['id']} run {n} ({result.agent_turns} turns, {result.seconds:.0f}s)", flush=True)
            return result

    results = await asyncio.gather(*(guarded(s, n) for s in scenarios for n in range(1, args.runs + 1)))
    report = write_report(scenarios, results, args.runs, base.llm_model)
    runs_dir = ROOT / "runs"
    runs_dir.mkdir(exist_ok=True)
    raw = runs_dir / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    raw.write_text(
        json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nreport: {report}\nraw:    {raw}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
