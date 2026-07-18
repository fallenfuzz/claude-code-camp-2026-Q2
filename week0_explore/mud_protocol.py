#!/usr/bin/env python3
"""
mud_protocol.py - shared telnet/ANSI protocol layer for tbaMUD tooling.

Pure protocol decoding: telnet IAC negotiation stripping, ANSI-aware line
buffering, and room-block recognition (title / exits / mob-sighting lines).
No world knowledge here - no vnums, no static world data, no game-specific
identity logic. Safe to import from any tool that plays or observes this
MUD server, regardless of what that tool does with the parsed lines
(journey_map's ground-truth cartography, challenge_agent's self-taught
exploration, etc).

Extracted unmodified (behavior-preserving move, not a rewrite) from
journey_map.py on 2026-07-17 so more than one tool can import it.
journey_map.py now imports TelnetStripper and SessionParser from here
instead of defining them inline; its own behavior is unchanged.
"""

import re

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
EXITS_RE = re.compile(r"^\[\s*Exits:\s*(.*?)\s*\]$")
# Observed live prompt: "20H 100M 80V (news) (motd) > " (no trailing newline).
PROMPT_RE = re.compile(r"\d+H\s+\d+M\s+\d+V\b.*>\s*$")

YELLOW = "\x1b[0;33m"
CYAN = "\x1b[0;36m"

MOVE_ALIASES = {
    "n": "n", "north": "n",
    "e": "e", "east": "e",
    "s": "s", "south": "s",
    "w": "w", "west": "w",
    "u": "u", "up": "u",
    "d": "d", "down": "d",
}


def normalize_text(s):
    """Strip ANSI codes and collapse whitespace, for line comparisons."""
    s = ANSI_RE.sub("", s or "")
    return " ".join(s.split()).strip()


# ---------------------------------------------------------------------------
# Telnet IAC stripper (text extraction only - never touches forwarded bytes)
# ---------------------------------------------------------------------------

IAC, DONT, DO, WONT, WILL, SB, SE = 255, 254, 253, 252, 251, 250, 240


class TelnetStripper:
    """Stateful (across chunks) telnet IAC negotiation stripper, used only to
    get clean text for parsing. A tap proxy forwards the original raw bytes
    untouched regardless of what this produces; a direct client (no proxy in
    the middle) uses this as its only decode step."""

    def __init__(self):
        self.state = "DATA"

    def feed(self, data: bytes) -> str:
        out = bytearray()
        for b in data:
            if self.state == "DATA":
                if b == IAC:
                    self.state = "IAC"
                else:
                    out.append(b)
            elif self.state == "IAC":
                if b == IAC:
                    out.append(IAC)
                    self.state = "DATA"
                elif b in (DO, DONT, WILL, WONT):
                    self.state = "NEG"
                elif b == SB:
                    self.state = "SB"
                else:
                    self.state = "DATA"
            elif self.state == "NEG":
                self.state = "DATA"
            elif self.state == "SB":
                if b == IAC:
                    self.state = "SB_IAC"
            elif self.state == "SB_IAC":
                if b == SE:
                    self.state = "DATA"
                else:
                    self.state = "SB"
        return bytes(out).decode("latin-1")


# ---------------------------------------------------------------------------
# Stream parser: turns server->client text into resolved room blocks
# ---------------------------------------------------------------------------

class SessionParser:
    """Per-session (one telnet connection through a tap proxy). Buffers
    server text into lines, recognizes a full room block (title?...exits...
    entity lines...prompt), and hands it to journey.observe(). Non-room
    output (failed moves, combat spam, async chatter, the login sequence)
    has no exits line and is silently ignored.

    `journey` is duck-typed: any object with an
    `observe(title, exit_letters, move_dir, mob_lines)` method works - this
    class has no dependency on any particular localizer implementation.

    Note: this class is tailored to journey_map's async tap-proxy shape
    (commands and replies arrive on separate byte streams, paired via
    `note_command`'s FIFO) and only surfaces room blocks (title/exits/mob
    lines) - it silently drops any reply block that has no Exits: line,
    which includes `list`/`examine`/`consider` output. A synchronous
    client that needs those replies captured (challenge_agent.py) reads
    full response blocks itself instead of reusing this class - see that
    file's MudClient for why."""

    def __init__(self, journey):
        self.journey = journey
        self.stripper = TelnetStripper()
        self.buf = ""
        self.pending_lines = []
        self.command_queue = []

    def note_command(self, cmd):
        cmd = cmd.strip().lower()
        if cmd:
            self.command_queue.append(cmd)

    def feed_server_bytes(self, data: bytes):
        text = self.stripper.feed(data)
        if not text:
            return
        self.buf += text
        while True:
            idx = self.buf.find("\n")
            if idx == -1:
                break
            raw_line = self.buf[:idx]
            self.buf = self.buf[idx + 1:]
            if raw_line.endswith("\r"):
                raw_line = raw_line[:-1]
            stripped = normalize_text(raw_line)
            if stripped:
                self.pending_lines.append((raw_line, stripped))

        # The prompt has no trailing newline - it is whatever is left in buf.
        remainder_stripped = normalize_text(self.buf)
        if remainder_stripped and PROMPT_RE.search(remainder_stripped):
            self.buf = ""
            self._resolve_block()

    def _resolve_block(self):
        lines = self.pending_lines
        self.pending_lines = []
        cmd = self.command_queue.pop(0) if self.command_queue else None

        exits_idx = None
        exit_letters = []
        for i, (raw, stripped) in enumerate(lines):
            if raw.startswith(CYAN):
                m = EXITS_RE.match(stripped)
                if m:
                    exits_idx = i
                    exit_letters = m.group(1).split()
                    break

        if exits_idx is None:
            # Not a room block: failed move, combat spam, tells/channels,
            # login/menu text, etc. State is untouched.
            return

        title = None
        for raw, stripped in lines[:exits_idx]:
            if raw.startswith(YELLOW):
                title = stripped
                break

        mob_lines = [stripped for raw, stripped in lines[exits_idx + 1:]
                     if raw.startswith(YELLOW)]

        move_dir = MOVE_ALIASES.get(cmd) if cmd else None
        self.journey.observe(title, exit_letters, move_dir, mob_lines)
