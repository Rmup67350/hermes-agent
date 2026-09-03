from __future__ import annotations

from unittest.mock import Mock


def test_models_dev_probe_switch_forces_cache_only(monkeypatch):
    monkeypatch.setenv("HERMES_DISABLE_EXTERNAL_PROBES", "1")
    import agent.models_dev as models_dev

    network = Mock(return_value=({"network": {}}, None))
    monkeypatch.setattr(models_dev, "_models_dev_cache", {})
    monkeypatch.setattr(models_dev, "_load_disk_cache", lambda: {"cached": {}})
    monkeypatch.setattr(models_dev, "_fetch_models_dev_from_network", network)

    assert models_dev.fetch_models_dev(force_refresh=True) == {"cached": {}}
    network.assert_not_called()


def test_openrouter_metadata_probe_switch_uses_disk_cache(monkeypatch):
    monkeypatch.setenv("HERMES_DISABLE_EXTERNAL_PROBES", "true")
    import agent.model_metadata as metadata

    request = Mock()
    monkeypatch.setattr(metadata, "_model_metadata_cache", {})
    monkeypatch.setattr(metadata, "_load_model_metadata_disk_cache", lambda: {"m": {}})
    monkeypatch.setattr(metadata, "_model_metadata_disk_cache_age_seconds", lambda: 10.0)
    monkeypatch.setattr(metadata.requests, "get", request)

    assert metadata.fetch_model_metadata(force_refresh=True) == {"m": {}}
    request.assert_not_called()


def test_environment_probe_switch_does_not_start_worker(monkeypatch):
    monkeypatch.setenv("HERMES_DISABLE_EXTERNAL_PROBES", "yes")
    import tools.env_probe as env_probe

    start = Mock()
    monkeypatch.setattr(env_probe, "_ensure_probe_started", start)

    assert env_probe.get_environment_probe_line(force_refresh=True) == ""
    start.assert_not_called()
