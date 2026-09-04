from hermes_cli.fallback_config import get_fallback_chain


NESTED_CONFIG = {
    "fallback_model": {
        "enable_fallback": True,
        "fallback_providers": [
            {"provider": "openai-codex", "model": "gpt-5.6-sol"},
            {"provider": "zai", "model": "glm-5"},
        ],
    },
}


def test_nested_fallback_providers_feed_the_chain():
    chain = get_fallback_chain(NESTED_CONFIG)
    assert [(e["provider"], e["model"]) for e in chain] == [
        ("openai-codex", "gpt-5.6-sol"),
        ("zai", "glm-5"),
    ]


def test_top_level_fallback_providers_keep_priority_over_nested():
    config = {
        "fallback_providers": [
            {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        ],
        **NESTED_CONFIG,
    }
    chain = get_fallback_chain(config)
    assert (chain[0]["provider"], chain[0]["model"]) == (
        "anthropic", "claude-sonnet-4-6",
    )
    assert [(e["provider"], e["model"]) for e in chain[1:]] == [
        ("openai-codex", "gpt-5.6-sol"),
        ("zai", "glm-5"),
    ]


def test_nested_duplicates_of_direct_entries_are_deduplicated():
    config = {
        "fallback_model": {
            "provider": "zai",
            "model": "glm-5",
            "fallback_providers": [
                {"provider": "zai", "model": "glm-5"},
                {"provider": "openai-codex", "model": "gpt-5.6-sol"},
            ],
        },
    }
    chain = get_fallback_chain(config)
    assert [(e["provider"], e["model"]) for e in chain] == [
        ("zai", "glm-5"),
        ("openai-codex", "gpt-5.6-sol"),
    ]


def test_disabled_nested_chain_is_ignored():
    config = {
        "fallback_model": {
            "enable_fallback": False,
            "fallback_providers": [
                {"provider": "zai", "model": "glm-5"},
            ],
        },
    }
    assert get_fallback_chain(config) == []
