from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch


def test_native_gemini_detection_requires_exact_google_hostname():
    from agent.gemini_native_adapter import is_native_gemini_base_url

    assert is_native_gemini_base_url(
        "https://generativelanguage.googleapis.com/v1beta"
    )
    for unsafe in (
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "https://generativelanguage.googleapis.com.evil.example/v1beta",
        "https://evil.example/generativelanguage.googleapis.com/v1beta",
        "http://generativelanguage.googleapis.com/v1beta",
        "https://user:pass@generativelanguage.googleapis.com/v1beta",
        "https://generativelanguage.googleapis.com:443/v1beta",
        "https://generativelanguage.googleapis.com/v1beta?alt=sse",
        "https://generativelanguage.googleapis.com/v1beta#fragment",
    ):
        assert not is_native_gemini_base_url(unsafe)


def test_native_gemini_abort_reaches_http_pool_without_closing_socket():
    from agent.agent_runtime_helpers import force_close_tcp_sockets
    from agent.gemini_native_adapter import GeminiNativeClient

    class FakeSocket:
        def __init__(self):
            self.shutdown_calls = 0
            self.close_calls = 0

        def shutdown(self, _how):
            self.shutdown_calls += 1

        def close(self):
            self.close_calls += 1

    sock = FakeSocket()
    stream = SimpleNamespace(_sock=sock)
    http11 = SimpleNamespace(_network_stream=stream)
    pool_entry = SimpleNamespace(_connection=http11)
    pool = SimpleNamespace(_connections=[pool_entry])
    transport = SimpleNamespace(_pool=pool)
    http_client = SimpleNamespace(_transport=transport)
    gemini_client = GeminiNativeClient(
        api_key="test-google-key",
        http_client=cast(Any, http_client),
    )

    assert force_close_tcp_sockets(gemini_client) == 1
    assert sock.shutdown_calls == 1
    assert sock.close_calls == 0


def test_google_alias_native_fallback_keeps_gemini_native_request_client():
    from agent.gemini_native_adapter import GeminiNativeClient
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-google-key",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        provider="google",
        model="gemini-3.5-flash",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    setattr(agent, "api_mode", "chat_completions")
    try:
        getattr(agent, "_client_kwargs")["ssl_verify"] = False
        keepalive_http = MagicMock()
        with patch.object(
            agent,
            "_build_keepalive_http_client",
            return_value=keepalive_http,
        ) as build_keepalive:
            request_client = agent._create_request_openai_client(reason="test")

        assert isinstance(request_client, GeminiNativeClient)
        build_keepalive.assert_called_once_with(
            "https://generativelanguage.googleapis.com/v1beta",
            verify=False,
        )
        assert request_client._http is keepalive_http
    finally:
        agent.close()
