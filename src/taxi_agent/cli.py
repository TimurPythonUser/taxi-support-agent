"""Command line: `taxi-agent chat` (in-process or against a deployed API) and `taxi-agent serve`."""

import argparse
import asyncio
import getpass
import json
import os
import sys
from typing import Any

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()


def _show_turn(reply: str, events: list[dict[str, Any]], missing: list[str], debug: bool) -> None:
    if debug:
        for e in events:
            label = f"[dim]{e['kind']}[/] [cyan]{e['name']}[/]"
            if e["kind"] == "tool":
                args = json.dumps(e["arguments"], ensure_ascii=False)
                ok = e["result"].get("ok")
                label += f" {args} → {'ok' if ok else e['result'].get('error', ok)}"
            console.print(label)
        if missing:
            console.print(f"[dim]missing: {', '.join(missing)}[/]")
    console.print(Panel(reply, title="агент", title_align="left", border_style="green"))


async def _chat_local(debug: bool) -> None:
    from dataclasses import asdict

    from taxi_agent.agent.state import CaseStatus
    from taxi_agent.config import get_settings
    from taxi_agent.factory import build_agent

    agent = build_agent(get_settings())
    state = await agent.start_session()
    console.print(f"[dim]session {state.session_id} (локально, SQLite в {agent.settings.db_path})[/]")
    while True:
        text = Prompt.ask("[bold]вы[/]").strip()
        if text.lower() in {"/q", "/quit", "exit"}:
            return
        if not text:
            continue
        with console.status("думаю..."):
            turn = await agent.handle_message(state.session_id, text)
        case = turn.state.active_case
        missing = case.missing_fields() if case and case.status == CaseStatus.COLLECTING else []
        _show_turn(turn.reply, [asdict(e) for e in turn.events], missing, debug)


def _chat_remote(url: str, user: str | None, password: str | None, debug: bool) -> None:
    if user and not password:
        password = os.environ.get("TAXI_AGENT_PASSWORD") or getpass.getpass(f"Password for {user}: ")
    auth = (user, password) if user else None
    with httpx.Client(base_url=url.rstrip("/"), auth=auth, timeout=90) as client:
        resp = client.post("/api/sessions")
        resp.raise_for_status()
        session_id = resp.json()["session_id"]
        console.print(f"[dim]session {session_id} ({url})[/]")
        while True:
            text = Prompt.ask("[bold]вы[/]").strip()
            if text.lower() in {"/q", "/quit", "exit"}:
                return
            if not text:
                continue
            with console.status("думаю..."):
                resp = client.post(f"/api/sessions/{session_id}/messages", json={"text": text})
            resp.raise_for_status()
            data = resp.json()
            _show_turn(data["reply"], data["events"], data["missing_fields"], debug)


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="taxi-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", help="chat with the agent in the terminal")
    chat.add_argument("--url", help="base URL of a running API; omit to run the agent in-process")
    chat.add_argument("--user", help="Basic Auth user for --url")
    chat.add_argument("--password", help="Basic Auth password (better: TAXI_AGENT_PASSWORD env or the prompt)")
    chat.add_argument("--debug", action="store_true", help="show tool calls and missing fields")

    serve = sub.add_parser("serve", help="run the HTTP API and web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "serve":
        import uvicorn

        uvicorn.run("taxi_agent.api.app:create_app", factory=True, host=args.host, port=args.port, reload=args.reload)
    elif args.url:
        _chat_remote(args.url, args.user, args.password, args.debug)
    else:
        asyncio.run(_chat_local(args.debug))


if __name__ == "__main__":
    main()
