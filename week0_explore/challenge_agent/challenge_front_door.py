#!/usr/bin/env python3
"""
challenge_front_door.py - the ONLY file in this codebase that talks to an
LLM. Two call sites total, both Haiku 4.5 (claude-haiku-4-5-20251001):

  1. interpret_challenge(text) - challenge text -> a known execution-engine
     primitive call, or `unsupported` with a short reason. The prompt below
     contains no world knowledge whatsoever: no map, no mob data, nothing
     about this specific game. Its only job is language -> primitive.

  2. phrase_report(report) - the engine's raw structured result -> a short
     natural-language reply. Optional and never load-bearing: if this call
     raises or the API is unavailable, the caller (challenge_agent.py's
     main()) falls back to printing the raw structured result directly.

Everything else in this codebase - exploration, room identity, shop/mob
detection, pathfinding, the report itself - is deterministic Python in
challenge_agent.py and never imports this module or `anthropic`.

Requires ANTHROPIC_API_KEY in the environment. Never hardcoded, never
committed (course automatic-failure condition bans committed env vars) -
see README.md.
"""

import json
import os

import anthropic

MODEL = "claude-haiku-4-5-20251001"

# The fixed list of primitives the execution engine currently supports.
# Deliberately hand-maintained here rather than introspected from the
# engine - keeps this module's only coupling to challenge_agent.py at the
# name level (see challenge_agent.PRIMITIVES), never at the implementation
# level, so the front door genuinely knows nothing about how a primitive
# works, only that it exists.
SUPPORTED_PRIMITIVES = [
    {
        "name": "find_place",
        "description": (
            "Explore the world until a room matching a place keyword is "
            "found (e.g. a bakery, a blacksmith, an inn), then report what "
            "is sold there and what is nearby."
        ),
        "params": {"keyword": "str - the kind of place being sought, e.g. 'bakery'"},
    },
]

INTERPRET_TOOL = {
    "name": "resolve_challenge",
    "description": (
        "Resolve a player's plain-language challenge into a known "
        "execution-engine primitive call, or mark it unsupported."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "supported": {
                "type": "boolean",
                "description": "True if the challenge maps to one of the known primitives.",
            },
            "primitive": {
                "type": "string",
                "enum": [p["name"] for p in SUPPORTED_PRIMITIVES],
                "description": "Required if supported is true.",
            },
            "params": {
                "type": "object",
                "description": "Params for the chosen primitive, e.g. {\"keyword\": \"bakery\"}.",
            },
            "reason": {
                "type": "string",
                "description": (
                    "Required if supported is false - a short, specific reason "
                    "why no known primitive covers this challenge."
                ),
            },
        },
        "required": ["supported"],
    },
}


def _client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it in your shell before "
            "running challenge_agent.py - see README.md."
        )
    return anthropic.Anthropic(api_key=api_key)


def interpret_challenge(challenge_text):
    """Call 1 of 2. Maps free-form challenge text to a primitive call.

    Returns either:
        {"supported": True, "primitive": "find_place", "params": {"keyword": "bakery"}}
    or:
        {"supported": False, "reason": "..."}
    """
    primitives_desc = "\n".join(
        f"- {p['name']}({', '.join(p['params'])}): {p['description']}"
        for p in SUPPORTED_PRIMITIVES
    )
    prompt = (
        "You translate a player's plain-language challenge for a MUD-playing "
        "agent into one of the agent's known primitive calls. You have no "
        "knowledge of this specific game world - no map, no mob list, no "
        "item list. Your only job is matching language to a known primitive.\n\n"
        f"Known primitives:\n{primitives_desc}\n\n"
        f"Challenge: {challenge_text!r}\n\n"
        "Call resolve_challenge. If the challenge clearly matches a known "
        "primitive, set supported=true with that primitive and its params. "
        "If it does not match any known primitive (e.g. it asks to fight, "
        "buy, or level up - none of which are implemented yet), set "
        "supported=false with a short, specific reason."
    )
    client = _client()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=300,
        tools=[INTERPRET_TOOL],
        tool_choice={"type": "tool", "name": "resolve_challenge"},
        messages=[{"role": "user", "content": prompt}],
    )
    for block in resp.content:
        if block.type == "tool_use":
            result = block.input
            if result.get("supported"):
                return {
                    "supported": True,
                    "primitive": result.get("primitive"),
                    "params": result.get("params") or {},
                }
            return {
                "supported": False,
                "reason": result.get("reason", "challenge did not map to a known primitive"),
            }
    return {"supported": False, "reason": "front door returned no tool call"}


def phrase_report(report):
    """Call 2 of 2 (optional). Turns the engine's raw structured result into
    a short natural-language reply. Never load-bearing for correctness -
    callers must fall back to the raw structured result (e.g.
    json.dumps(report)) if this raises or the API is unavailable."""
    prompt = (
        "Write a short (3-5 sentence), plain-English summary of this MUD "
        "exploration result for the player who asked for it. Be concrete "
        "about what was found and where, using only the facts given below - "
        "do not invent details that are not in the structured result.\n\n"
        f"Structured result:\n{json.dumps(report, indent=2)}"
    )
    client = _client()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = [block.text for block in resp.content if block.type == "text"]
    return "".join(parts).strip()
