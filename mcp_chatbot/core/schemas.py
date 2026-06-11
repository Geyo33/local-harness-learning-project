from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base for structured-output models. ``extra="forbid"`` makes
    ``model_json_schema()`` emit ``additionalProperties: false``, so a backend
    in strict mode rejects responses padded with undeclared keys (and remote
    strict APIs accept the schema at all). Inherit this for any model fed to an
    LLM via ``response_format``."""

    model_config = ConfigDict(extra="forbid")


class EpisodeResult(StrictModel):
    """Shutdown co-extraction result: one structured object yielding both the
    episode summary and candidate durable facts. ``facts`` are plain strings —
    tags are deferred, the model never emits them."""

    summary: str
    facts: list[str]
