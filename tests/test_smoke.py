from rag_logic import build_chat_history
from langchain_core.messages import HumanMessage,AIMessage
messages = [{"role":"user", "content":"hello"}, {"role":"assistant","content":"hey"}]

def test_build_chat_history():
    history = build_chat_history(messages)
    assert len(history) == 2
    assert history[0].content == 'hello'
    assert isinstance(history[0], HumanMessage)
    assert history[1].content == 'hey'
    assert isinstance(history[1], AIMessage)
    for msg in messages:
        assert msg['role'] in ("user","assistant")