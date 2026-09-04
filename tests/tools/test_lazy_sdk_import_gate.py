"""Runtime SDK imports must not depend on package metadata."""

from __future__ import annotations

import importlib.util
import sys
import types

import pytest

from tools.lazy_deps import FeatureUnavailable

_REAL_FIND_SPEC = importlib.util.find_spec


@pytest.fixture
def blocked_lazy_installs(monkeypatch):
    def refuse(feature, *, prompt=True):
        raise FeatureUnavailable(
            feature,
            ("some-pkg==1.0",),
            "lazy installs disabled (security.allow_lazy_installs=false)",
        )

    import tools.lazy_deps as lazy_deps

    monkeypatch.setattr(lazy_deps, "ensure", refuse)


def _install_double(monkeypatch, name: str, **attrs):
    double = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(double, key, value)
    monkeypatch.setitem(sys.modules, name, double)
    return double


def test_import_fal_client_uses_an_importable_sdk(monkeypatch, blocked_lazy_installs):
    double = _install_double(monkeypatch, "fal_client", subscribe=lambda *a, **k: None)

    from tools.fal_common import import_fal_client

    assert import_fal_client() is double


def test_import_fal_client_still_raises_when_the_sdk_is_absent(
    monkeypatch, blocked_lazy_installs
):
    monkeypatch.delitem(sys.modules, "fal_client", raising=False)
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name, *a, **k: (
            None if name == "fal_client" else _REAL_FIND_SPEC(name, *a, **k)
        ),
    )

    from tools.fal_common import import_fal_client

    with pytest.raises(ImportError, match="lazy installs disabled"):
        import_fal_client()


def test_parallel_clients_use_an_importable_sdk(monkeypatch, blocked_lazy_installs):
    class Parallel:
        def __init__(self, api_key):
            self.api_key = api_key

    _install_double(monkeypatch, "parallel", Parallel=Parallel, AsyncParallel=Parallel)
    monkeypatch.setenv("PARALLEL_API_KEY", "test-key")

    import tools.web_tools as web_tools

    monkeypatch.setattr(web_tools, "_parallel_client", None, raising=False)
    monkeypatch.setattr(web_tools, "_async_parallel_client", None, raising=False)

    from plugins.web.parallel.provider import _get_async_client, _get_sync_client

    assert _get_sync_client().api_key == "test-key"
    assert _get_async_client().api_key == "test-key"
