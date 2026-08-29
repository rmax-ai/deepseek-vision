"""Preset model definition (imported by preset submodules and __init__)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Preset(BaseModel):
    """A named bundle of system prompt, instructions, and schema."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    system_prompt: str
    task_instructions: str
    output_schema: type[BaseModel] | None = None
    temperature: float | None = None
    max_output_tokens: int = 8192
    hierarchical: bool = False
    batch_instructions: str | None = None
    final_output_schema: type[BaseModel] | None = None
