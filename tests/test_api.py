import pytest
from fastapi.testclient import TestClient

from taxi_agent.api.app import create_app

from .helpers import call, say


@pytest.fixture
def client(settings, agent):
    with TestClient(create_app(settings=settings, agent=agent)) as c:
        yield c


def test_healthz(client):
    assert client.get("/healthz").json()["status"] == "ok"


def test_index_is_served(client):
    resp = client.get("/")
    assert resp.status_code == 200 and "text/html" in resp.headers["content-type"]


def test_conversation_over_http(client, fake_llm):
    session = client.post("/api/sessions")
    assert session.status_code == 201
    sid = session.json()["session_id"]

    fake_llm.extend(call("update_case", scenario="trip_price", origin="Крещатик 22"), say("Куда поедем?"))
    turn = client.post(f"/api/sessions/{sid}/messages", json={"text": "Сколько стоит от Крещатика 22?"}).json()
    assert turn["reply"] == "Куда поедем?"
    assert turn["missing_fields"] == ["destination"]
    assert turn["events"][0]["name"] == "update_case"
    assert turn["state"]["cases"][0]["origin"] == "Крещатик 22"

    history = client.get(f"/api/sessions/{sid}").json()
    assert [m["role"] for m in history["messages"]] == ["user", "assistant"]  # tool traffic is hidden


def test_unknown_session_is_404(client):
    assert client.post("/api/sessions/nope/messages", json={"text": "hi"}).status_code == 404
    assert client.get("/api/sessions/nope").status_code == 404


@pytest.mark.parametrize("text", ["", "x" * 2001])
def test_message_validation(client, text):
    sid = client.post("/api/sessions").json()["session_id"]
    assert client.post(f"/api/sessions/{sid}/messages", json={"text": text}).status_code == 422


def test_approach_page_404_when_not_built(client, settings, tmp_path):
    settings.approach_page = tmp_path / "missing.html"
    assert client.get("/approach").status_code == 404
