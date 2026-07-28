from rag_logic import build_chat_history, check_ollama_heath
from langchain_core.messages import HumanMessage,AIMessage
import pytest
import requests

messages = [{"role":"user", "content":"hello"},
                {"role": "system", "content": "you are a bot"},   # unknown role
             {"role":"assistant","content":"hey"}]


class FakeResponse:
    def __init__(self, data):
        self._data = data
    def raise_for_status(self):
        pass                      # success = do nothing
    def json(self):
        return self._data



@pytest.mark.parametrize("role, expected_type", [("user", HumanMessage), ("assistant", AIMessage)])
def test_role_to_type(role, expected_type):
    history = build_chat_history([{"role":role, "content":"xyz"}])
    assert isinstance(history[0], expected_type)

def test_build_chat_history():
    history = build_chat_history(messages)
    assert len(history) == 2
    assert [m.content for m in history] == ["hello", "hey"]

def test_ollama_ready(monkeypatch):
    # Arrange: fake requests.get to return a response listing llama3.2
    def fake_get(url, timeout):
        return FakeResponse({"models": [{"name": "llama3.2"}]})
    monkeypatch.setattr(requests, "get", fake_get)

    # Act
    ok, message = check_ollama_heath()

    # Assert
    assert ok is True

def test_model_not_pulled(monkeypatch):
    def fake_get(url,timeout):
        return FakeResponse({"models": [{"name": "mistral"}]})
    monkeypatch.setattr(requests, "get", fake_get)

    ok, message = check_ollama_heath()

    assert ok is False
    assert "isn't pulled" in message

def test_ollama_down(monkeypatch):
    def fake_get(url, timeout):
        raise  requests.exceptions.RequestException
    monkeypatch.setattr(requests, "get", fake_get)

    ok, message = check_ollama_heath()

    assert ok is False
    assert "doesn't seem to be running" in message