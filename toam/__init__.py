"""TOAM: a tiny model for tool use and agent control."""

from .inference import TinyAgent, format_prompt, parse_action

__all__ = ["TinyAgent", "format_prompt", "parse_action"]
