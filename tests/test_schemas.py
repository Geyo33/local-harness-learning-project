import pytest
from pydantic import ValidationError

from mcp_chatbot.core.schemas import EpisodeResult, StrictModel


def test_strict_model_forbids_extra():
    class M(StrictModel):
        a: int

    M(a=1)  # declared field OK
    with pytest.raises(ValidationError):
        M(a=1, b=2)  # undeclared field rejected


def test_episode_result_schema_has_additional_properties_false():
    schema = EpisodeResult.model_json_schema()
    assert schema["additionalProperties"] is False


def test_episode_result_parses_valid_payload():
    data = EpisodeResult.model_validate_json(
        '{"summary": "did stuff", "facts": ["one", "two"]}'
    )
    assert data.summary == "did stuff"
    assert data.facts == ["one", "two"]


def test_episode_result_rejects_extra_key():
    with pytest.raises(ValidationError):
        EpisodeResult.model_validate_json(
            '{"summary": "s", "facts": [], "mood": "happy"}'
        )
