#!/usr/bin/env python3
"""
challenge_agent.py - deterministic execution engine for the challenge agent.

This module contains ZERO AI. It connects to the tbaMUD test server (through
journey_map's tap proxy on :4001, per the architecture doc), plays like a
real player would - look, move, exits, examine, consider, list - and builds
its own world model purely from what it observes. It never opens any file
under `preview/data/world/` (that is journey_map's ground-truth data, used
for a different, non-agentic job - faithful cartography of already-visited
rooms; this agent is deliberately self-taught instead).

The only two places in this whole codebase that call an LLM are in
`challenge_front_door.py` - this file never imports `anthropic` and never
sees an API key. See that file's docstring and this project's README for
the exact boundary.

Run: `python3 challenge_agent.py "find the bakery"` (needs ANTHROPIC_API_KEY
in the environment - see README). Progress persists to `agent_memory.json`
(gitignored) so a killed/restarted run resumes instead of starting cold.
"""

import argparse
import json
import os
import re
import socket
import sys
import time
from collections import deque
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MEMORY_PATH = BASE_DIR / "agent_memory.json"

# mud_protocol.py is shared with journey_map/ (see that file's docstring for
# what was extracted and why - pure telnet/ANSI protocol decoding, no world
# knowledge). It lives one directory up, at week0_explore/.
sys.path.insert(0, str(BASE_DIR.parent))
from mud_protocol import (  # noqa: E402
    EXITS_RE, PROMPT_RE, TelnetStripper, YELLOW, CYAN, normalize_text,
)

# Connect through journey_map's tap proxy, not directly to the MUD, so its
# viewer sees this agent's exploration for free (architecture doc). Known
# limitation inherited from journey_map: only one active session through the
# proxy at a time.
PROXY_HOST = "localhost"
PROXY_PORT = 4001

DIRS = ["n", "e", "s", "w", "u", "d"]
OPPOSITE_DIR = {"n": "s", "s": "n", "e": "w", "w": "e", "u": "d", "d": "u"}
DIR_WORD_TO_LETTER = {
    "north": "n", "east": "e", "south": "s", "west": "w", "up": "u", "down": "d",
}

# Small, fixed synonym table for find_place's keyword family. Milestone 1
# scope is bakery-only (architecture doc's "open question" flagged this -
# confirmed with the Orchestrator to start with bakery only; grows via this
# dict, not user-configurable yet).
KEYWORD_FAMILIES = {
    "bakery": ["bakery", "baker", "bread", "oven"],
}

# Confirmed live against the running tbaMUD 2025 container (dev-time probe,
# cross-checked against tbamud/tbamud source: act.informative.c do_examine /
# look_at_target, fight.c do_consider, shop.c shopping_list).
SHOP_FAIL_TEXT = "cannot do that here"
EXAMINE_FAIL_TEXT = "you do not see that here"
CONSIDER_FAIL_TEXT = "consider killing who"
# Distinguishes "genuinely blocked" ("Alas, you cannot go that way...") from
# "movement worked but the destination is dark, so no room block was
# printed" ("It is pitch black..."). Both make parse_room_block() return
# None - conflating them corrupts blocked_exits with directions that
# actually work fine, discovered live 2026-07-17 during the DFS frontier
# explorer's first clean-pocket run (see step()/_grope_for_light()).
DARK_TEXT = "pitch black"

# Rest/stamina management (architecture_v2.md Frontier explorer Amendment 2
# addendum, 2026-07-17 - promoted from accepted milestone-1 gap to required):
# exhaustive DFS-with-backtracking takes far more steps to fully clear a
# pocket than greedy nearest-first ever did, so running out of movement
# mid-exploration is now the normal, expected thing on any real run, not a
# rare edge case a human works around between test sessions. "You are too
# exhausted" is the server's reactive signal (a move failed on cost even
# though the last-known prompt hadn't yet shown 0 - the prompt only updates
# after a command completes); a movement value of 0 in the prompt/score is
# the proactive signal. See needs_rest()/rest_cycle_next_action()/
# Agent.rest_cycle().
EXHAUSTED_TEXT = "you are too exhausted"
REST_TRIGGER_MOVES = 0
REST_MIN_MOVES = 30  # "not necessarily full - just enough to resume productively"
REST_POLL_INTERVAL_SECONDS = 10  # real wall-clock delay between score polls
REST_MAX_POLLS = 80  # safety cap (~13 minutes) in case regen/parsing stalls

PLAYER_STATE_PROMPT_RE = re.compile(r"(\d+)H\s+(\d+)M\s+(\d+)V")


def parse_player_state_from_prompt(prompt_text):
    """Pure: every tbaMUD prompt ends with "<n>H <n>M <n>V ..." (HP/Mana/
    Movement) - already used as the block-terminator sentinel
    (mud_protocol.PROMPT_RE), but the numbers themselves were previously
    discarded. Returns {"hp","mana","moves"} or None if prompt_text doesn't
    contain the pattern (e.g. empty string - no prompt captured yet)."""
    if not prompt_text:
        return None
    m = PLAYER_STATE_PROMPT_RE.search(prompt_text)
    if not m:
        return None
    return {"hp": int(m.group(1)), "mana": int(m.group(2)), "moves": int(m.group(3))}


def parse_score(text):
    """Pure: parse the `score` command's free-form reply into a flat dict of
    whatever fields are present. Tolerant by design - a field's regex
    simply not matching just omits that key, never raises; live tbaMUD
    score output can vary a bit by class/level/immortal status. Groundwork
    for primitives beyond find_place (leveling, buying, combat all need
    level/gold/HP) - same live-text-only, zero-AI pattern as every other
    heuristic in this file."""
    state = {}
    m = re.search(r"(\d+)\((\d+)\)\s*hit", text)
    if m:
        state["hp"], state["hp_max"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+)\((\d+)\)\s*mana", text)
    if m:
        state["mana"], state["mana_max"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+)\((\d+)\)\s*movement points", text)
    if m:
        state["moves"], state["moves_max"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"armor class is (-?\d+)/(-?\d+)", text)
    if m:
        state["armor_class"], state["armor_class_alt"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"alignment is (-?\d+)", text)
    if m:
        state["alignment"] = int(m.group(1))
    m = re.search(r"You have (\d+) exp,", text)
    if m:
        state["exp"] = int(m.group(1))
    m = re.search(r"(\d+) gold coins", text)
    if m:
        state["gold"] = int(m.group(1))
    m = re.search(r"(\d+) questpoints", text)
    if m:
        state["questpoints"] = int(m.group(1))
    m = re.search(r"You need (\d+) exp to reach your next level", text)
    if m:
        state["exp_to_next_level"] = int(m.group(1))
    m = re.search(r"ranks you as (.+?)\s*\(level (\d+)\)", text)
    if m:
        state["title"], state["level"] = m.group(1).strip(), int(m.group(2))
    m = re.search(r"You are (standing|sleeping|resting|sitting|fighting)\b", text, re.IGNORECASE)
    if m:
        state["posture"] = m.group(1).lower()
    return state


def needs_rest(player_state, exhausted_signal=False, trigger_moves=REST_TRIGGER_MOVES):
    """Pure: should a rest cycle begin? Fires on either signal the
    architecture doc specifies: the last-known movement value has hit
    `trigger_moves` (default 0), or the server just said "You are too
    exhausted" on the last command (the reactive fallback - the prompt only
    updates after a command completes, so a move that costs more than the
    remaining movement can fail while the last-seen prompt still showed a
    small positive number)."""
    if exhausted_signal:
        return True
    moves = (player_state or {}).get("moves")
    if moves is None:
        return False
    return moves <= trigger_moves


def rest_cycle_next_action(phase, moves, min_moves=REST_MIN_MOVES):
    """Pure state-machine step for the rest cycle: start -> rest ->
    poll(score) until moves >= min_moves -> stand -> done.

    Uses `rest` rather than `sleep` (live-confirmed 2026-07-17 against the
    running tbaMUD 2025 container: `rest` needs no prerequisite `sit`, and
    exits cleanly with a single `stand` - "You stop resting, and stand up" -
    no `wake` step, no risk of the character being asleep mid-cycle if
    something interrupts it. tbaMUD also has `sleep`, a deeper/faster-regen
    state that requires `wake` before `stand` - not used here since `rest`
    is simpler and sufficient for the modest partial-recovery threshold
    this cycle targets; revisit if a future primitive needs faster regen
    badly enough to be worth the extra step).

    Kept fully separate from any real time.sleep() so it is unit-testable
    without live time - the caller (Agent.rest_cycle()) supplies
    phase/moves and owns the actual wall-clock delay between "score" polls,
    and the real command dispatch. Returns (next_phase, action) where
    action is one of "rest"/"score"/"stand"/None (nothing left to send -
    the cycle is complete)."""
    if phase == "start":
        return "resting", "rest"
    if phase == "resting":
        if moves is not None and moves >= min_moves:
            return "standing", "stand"
        return "resting", "score"
    return "done", None


# Heuristic noun/keyword extraction stopwords - deliberately small and
# generic (articles, prepositions, pronouns, common presence-verbs). A false
# positive here just costs one harmless "you do not see that here" /
# "Consider killing who?" round trip, so this list does not need to be
# exhaustive; same accepted-imprecision spirit as the localizer's
# over-splitting behavior (a curious real player guesses at keywords the
# same way, and would rather assume "new room" than confidently misjudge
# "same room" from a glance).
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "here", "there", "and",
    "of", "in", "on", "at", "to", "with", "from", "near", "you", "your",
    "this", "that", "it", "he", "she", "they", "stands", "stand", "sits",
    "sit", "standing", "sitting", "looking", "looks", "walks", "walking",
    "some", "into", "onto", "upon", "who", "what", "his", "her", "its",
}
ARTICLE_NOUN_RE = re.compile(r"\b(?:a|an|some)\s+([a-z]+)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# MudClient - a synchronous telnet CLIENT (not a tap proxy).
#
# journey_map's SessionParser is deliberately NOT reused here even though it
# is "protocol layer" code: it is built for an async tap (commands and
# replies arrive on separate byte streams, paired by a FIFO in
# note_command/_resolve_block) and it only ever surfaces room blocks - any
# reply with no "Exits:" line (which includes every list/examine/consider
# response) is silently dropped by its _resolve_block(). This agent needs
# those non-room replies captured in full, and it is a single synchronous
# actor (send one command, wait for its own reply, decide the next command)
# rather than a passive observer of someone else's session. So this class
# reuses the shared low-level pieces (TelnetStripper, normalize_text,
# PROMPT_RE, the color/exits-line constants) but implements its own simple
# accumulate-until-prompt reader instead of SessionParser. Flagged in the
# implementer report as a deliberate deviation from "reuse SessionParser".
# ---------------------------------------------------------------------------

class MudClient:
    def __init__(self, username, password, host=PROXY_HOST, port=PROXY_PORT):
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.sock = None
        self.stripper = TelnetStripper()
        self.buf = ""
        # The trailing prompt line (e.g. "23H 100M 85V (news) (motd) >")
        # that terminates every block - read_block() matches it to know a
        # reply is complete but historically discarded the text itself.
        # Captured here so callers can parse HP/Mana/Movement out of it
        # (architecture_v2.md addendum, 2026-07-17: continuous player-state
        # tracking) without changing read_block()'s return contract.
        self.last_prompt_text = ""

    def connect(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=15)
        self.sock.settimeout(10)

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass

    def _send(self, line):
        self.sock.sendall((line + "\r\n").encode("latin-1"))

    def _recv_chunk(self):
        try:
            data = self.sock.recv(4096)
        except socket.timeout:
            return ""
        if not data:
            raise ConnectionError("MUD connection closed")
        return self.stripper.feed(data)

    def read_block(self, budget=10.0):
        """Accumulate lines until the trailing prompt appears in the
        unterminated remainder (same detection rule as journey_map's
        SessionParser). Returns a list of (raw_line, stripped_line) tuples
        for the *entire* reply - unlike SessionParser, nothing is filtered
        out here; callers decide what matters."""
        end = time.time() + budget
        lines = []
        while time.time() < end:
            self.buf += self._recv_chunk()
            while True:
                idx = self.buf.find("\n")
                if idx == -1:
                    break
                raw = self.buf[:idx]
                self.buf = self.buf[idx + 1:]
                if raw.endswith("\r"):
                    raw = raw[:-1]
                stripped = normalize_text(raw)
                if stripped:
                    lines.append((raw, stripped))
            remainder = normalize_text(self.buf)
            if remainder and PROMPT_RE.search(remainder):
                self.last_prompt_text = remainder
                self.buf = ""
                return lines
        return lines

    def command(self, cmd, budget=10.0):
        """Send one command, wait for its full reply block, return it. This
        is the only way the engine talks to the MUD - single actor,
        synchronous request/response, no concurrent readers/writers."""
        self._send(cmd)
        return self.read_block(budget=budget)

    def login(self):
        """Login dance per this project's Runbook, confirmed live 2026-07-17
        against the running tbaMUD 2025 container: wait "By what name" (do
        not send anything before this appears - protocol detection eats
        early input), send name, wait "Password:", send password, wait
        "*** PRESS RETURN:", send empty line, wait "Make your choice:", send
        "1". If a prior session did not `quit` cleanly, reconnecting can
        drop straight into the game with no menu - this loop branches on
        whichever marker text actually appears next, in any order, rather
        than assuming a fixed sequence, so both cases are handled. The
        no-menu branch (immediate PROMPT_RE match with no login markers) is
        implemented per the Runbook but was not itself exercised live this
        session (every login observed here did show the menu, since prior
        test connections were closed cleanly) - flagged as
        implemented-but-not-live-verified."""
        buf = ""
        end = time.time() + 20
        while time.time() < end:
            buf += self._recv_chunk()
            norm = normalize_text(buf).lower()
            if "by what name" in norm:
                self._send(self.username)
                buf = ""
            elif "password:" in norm:
                self._send(self.password)
                buf = ""
            elif "press return" in norm:
                self._send("")
                buf = ""
            elif "make your choice" in norm:
                self._send("1")
                buf = ""
            elif PROMPT_RE.search(normalize_text(buf)):
                # Either the post-choice welcome block, or (unclean-reconnect
                # case) we were dropped straight into the game. Either way,
                # discard this buffer and let the caller send an explicit
                # "look" through the normal command() path - simpler and
                # more robust than trying to parse this transitional text as
                # a room block, at the cost of one harmless extra round trip.
                self.buf = ""
                return
        raise RuntimeError(f"login timed out; last buffer: {buf[-300:]!r}")


# ---------------------------------------------------------------------------
# Room-block parsing (own implementation - see MudClient docstring for why
# SessionParser's filtering does not fit here). Same title/exits/mob-line
# detection rule as SessionParser (YELLOW title before Exits:, CYAN
# Exits: line, YELLOW mob lines after it) - kept behaviorally identical to
# journey_map's proven heuristic - but additionally keeps the plain-text
# room description lines, which this agent needs for keyword matching and
# noun extraction and which SessionParser discards.
# ---------------------------------------------------------------------------

def parse_room_block(lines):
    """lines: list of (raw, stripped) tuples for one full reply block.
    Returns a dict with title/exit_letters/desc_lines/mob_lines, or None if
    this reply has no Exits: line (a failed move, combat spam, the reply to
    list/examine/consider, etc - not a room block)."""
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
        return None

    title = None
    title_idx = None
    for i, (raw, stripped) in enumerate(lines[:exits_idx]):
        if raw.startswith(YELLOW):
            title = stripped
            title_idx = i
            break

    desc_lines = [
        stripped for i, (raw, stripped) in enumerate(lines[:exits_idx])
        if title_idx is None or i > title_idx
    ]
    mob_lines = [stripped for raw, stripped in lines[exits_idx + 1:]
                 if raw.startswith(YELLOW)]

    return {
        "title": title,
        "exit_letters": exit_letters,
        "desc_lines": desc_lines,
        "mob_lines": mob_lines,
    }


def extract_notable_nouns(desc_lines, max_n=5):
    """Heuristic: words following 'a'/'an'/'some' in the room description
    are usually the objects a curious player would examine (tbaMUD's extra
    descriptions are conventionally keyed by such nouns - confirmed live:
    `examine wall` hit a room's extra_desc keyed off "...ancient wall
    paintings..." in its description). Imprecise by nature (same accepted
    limitation as everywhere else guessing is involved) - a miss just costs
    one harmless failed `examine`."""
    text = " ".join(desc_lines)
    found = []
    seen = set()
    for m in ARTICLE_NOUN_RE.finditer(text):
        word = m.group(1).lower()
        if word in STOPWORDS or len(word) < 3:
            continue
        if word not in seen:
            seen.add(word)
            found.append(word)
        if len(found) >= max_n:
            break
    return found


def extract_target_candidates(mob_line, max_n=4):
    """Heuristic keyword guesses for `consider <target>`: the agent only
    ever sees a mob's room-presence long_desc line (e.g. "A baker stands
    here, kneading dough."), never its actual interact keyword list (it
    never has vnums/mob data) - so, like a real player, it just tries
    plausible words from what it read until the game recognizes one."""
    words = re.findall(r"[A-Za-z]+", mob_line.lower())
    found = []
    seen = set()
    for w in words:
        if w in STOPWORDS or len(w) < 3:
            continue
        if w not in seen:
            seen.add(w)
            found.append(w)
        if len(found) >= max_n:
            break
    return found


# ---------------------------------------------------------------------------
# Self-taught localizer (architecture doc, Components #5)
# ---------------------------------------------------------------------------

def identify_room(rooms, title, exit_letters, last_room_id, move_dir,
                   prev_move_dir=None, two_ago_room_id=None):
    """Pure function: given the agent's own known_rooms graph and one fresh
    observation, decide which room this is.

    Returns (room_id_or_None, is_new, same_text_matches).

    Amendment 2 (2026-07-17, immediate-backtrack recognition - checked
    right after Amendment 1's confirmed-edge fast path, before falling
    through to "always create a new node"): Amendment 1's graph-consistency
    -gated rule (below) turned out
    too strict for one specific, common case - a tight two-room
    bidirectional pair (e.g. street <-> shop, walked east then west) can
    NEVER graph-confirm its own return trip, because the reverse-direction
    edge is by definition being walked for the first time too. Every
    ping-pong between the two rooms minted a fresh phantom node forever
    (live-reproduced: 189 rooms discovered, 187 of them duplicates of just
    2 real rooms).

    Fix: recognize immediate backtracking specifically. If `move_dir` is
    the exact geometric opposite of `prev_move_dir` (the direction walked
    on the immediately preceding step) AND this observation's signature
    matches the room occupied *two steps ago* (`two_ago_room_id`, i.e.
    before that preceding step), treat it as that exact room. This only
    fires for a literal one-step-there-one-step-back reversal - it does
    NOT merge rooms reached by any other path, so the original bug (a
    *forward*, non-reversing chain of 3+ look-alike rooms) stays fixed: a
    genuine forward chain never has `move_dir == opposite(prev_move_dir)`
    two consecutive steps in a row, so this check simply never fires for
    it. `prev_move_dir`/`two_ago_room_id` are supplied by the caller
    (Agent tracks them across calls - see `_register_block`); both None
    for a context-free observation (e.g. a bootstrap look), so this check
    is a no-op there.

    Amendment 1 (2026-07-17, post-implementation finding - architecture_v2.md
    Localizer section): the merge condition is **graph-consistency-gated**,
    not text-uniqueness-gated. A room is only ever treated as "the same
    room we already know" (is_new=False) if the specific edge
    (last_room_id, move_dir) is *already recorded* in memory and points at
    a room whose signature matches what was just observed. If that exact
    edge has never been walked before, this is ALWAYS a brand-new node -
    even if exactly one (or more) known rooms happen to share the same
    (title, exit_letters) in text.

    The original rule ("exactly one text match -> merge") had a real bug,
    not just an accepted-ambiguity edge case: live-tested and reproduced
    twice (two separate runs, each hitting 3+ identically-titled,
    identically-exited rooms in a row - e.g. "The Great Field Of
    Midgaard"). The *first* repeat of such a run silently merged into the
    first-seen room, because it was the only known text match at that
    point - it never reached the old "2+ candidates -> tentative" path, so
    it looked "certain" by the letter of the old rule while actually being
    wrong. That silently corrupted the graph (a bogus edge added, a real
    second room never created) and capped how much of the world the
    frontier explorer could ever reach.

    This fix trades false-merge risk for over-splitting risk (two graph
    nodes for what might really be the same room) - accepted, since a real
    player would also default to "is this the same field, or a different
    one?" uncertainty rather than confidently assuming identity, and
    over-splitting is self-correcting (a little redundant exploration)
    where false-merging is not (a corrupted, silently-wrong map). Nodes
    created this way MAY be merged retroactively later if corroborating
    evidence appears (e.g. probe_reverse_edge() finds both nodes lead back
    to the same already-known neighbor by the same direction) - that
    retroactive-merge step is a should-have per the architecture doc, not
    implemented in this pass (see README/handoff notes).

    same_text_matches lists the room_ids that share (title, exit_letters)
    with this observation - always empty for a graph-confirmed match
    (nothing to disambiguate), non-empty when a new node is created despite
    looking textually identical to existing room(s) - purely informational/
    telemetry, never blocks node creation."""
    exit_key = tuple(sorted(exit_letters))

    # The ONLY way to resolve to an existing node: the specific edge just
    # walked is already recorded, and its recorded target's signature
    # matches what we just observed.
    if move_dir and last_room_id is not None and last_room_id in rooms:
        expected = rooms[last_room_id]["exits"].get(move_dir)
        if expected is not None and expected in rooms:
            candidate = rooms[expected]
            if (candidate["title"] == title
                    and tuple(sorted(candidate["exit_letters"])) == exit_key):
                return expected, False, []

    # Amendment 2: a literal one-step-there-one-step-back reversal. Only
    # fires when the direction just walked is the exact geometric opposite
    # of the direction walked on the immediately preceding step - a genuine
    # forward chain of look-alike rooms never satisfies this (each step in
    # a forward walk shares the same or an unrelated direction, never the
    # opposite of the step before it), so Amendment 1's fix stays intact.
    if (move_dir and prev_move_dir and two_ago_room_id is not None
            and two_ago_room_id in rooms
            and OPPOSITE_DIR.get(move_dir) == prev_move_dir):
        origin = rooms[two_ago_room_id]
        if (origin["title"] == title
                and tuple(sorted(origin["exit_letters"])) == exit_key):
            return two_ago_room_id, False, []

    same_text_matches = [
        rid for rid, r in rooms.items()
        if r["title"] == title and tuple(sorted(r["exit_letters"])) == exit_key
    ]
    return None, True, same_text_matches


class Agent:
    def __init__(self, client, memory):
        self.client = client
        self.memory = memory

    def save(self):
        tmp = str(MEMORY_PATH) + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.memory, fh, indent=2)
        os.replace(tmp, MEMORY_PATH)

    def _new_room_id(self):
        rid = str(self.memory["next_id"])
        self.memory["next_id"] += 1
        return rid

    def _cmd(self, cmd, budget=10.0):
        """The one funnel every command sent to the MUD goes through -
        movement, exits, examine, consider, list, look, score, sleep, wake,
        stand, all of it. Beyond sending/receiving, this is where continuous
        player-state tracking happens (architecture_v2.md addendum,
        2026-07-17): every reply's trailing prompt carries HP/Mana/Movement,
        parsed into memory["player_state"] on every single call, and a
        `score` reply additionally gets parsed for the fuller picture
        (level, gold, XP, armor class, alignment, etc). Pure live-text
        capture, zero AI, same pattern as every other heuristic in this
        file - not new decision logic, just data collection alongside
        whatever the caller actually asked for."""
        lines = self.client.command(cmd, budget=budget)
        prompt_state = parse_player_state_from_prompt(
            getattr(self.client, "last_prompt_text", ""))
        if prompt_state:
            self.memory.setdefault("player_state", {}).update(prompt_state)
        if cmd.strip().lower() == "score":
            text = " ".join(stripped for raw, stripped in lines)
            score_state = parse_score(text)
            if score_state:
                self.memory.setdefault("player_state", {}).update(score_state)
        return lines

    def rest_cycle(self, min_moves=REST_MIN_MOVES,
                    poll_interval=REST_POLL_INTERVAL_SECONDS,
                    max_polls=REST_MAX_POLLS):
        """Automatic rest cycle (architecture_v2.md Frontier explorer
        Amendment 2 addendum, 2026-07-17 - promoted from accepted gap to
        required): rest -> poll `score` until movement has recovered to at
        least `min_moves` (not necessarily full) -> stand -> resume. Uses
        `rest` rather than `sleep` (see rest_cycle_next_action()'s docstring
        for why - no wake step, no risk of staying asleep mid-cycle).
        Real-time wait (movement regenerates over game ticks) - this adds
        real wall-clock time to a run, which is expected, not a bug. Drives
        rest_cycle_next_action() (the pure state machine, unit-tested
        separately without live time) - this method owns only the actual
        command dispatch and the real time.sleep() between polls."""
        phase = "start"
        moves = self.memory.get("player_state", {}).get("moves")
        polls = 0
        while True:
            phase, action = rest_cycle_next_action(phase, moves, min_moves=min_moves)
            if action is None:
                break
            self._cmd(action)
            if action == "score":
                polls += 1
                moves = self.memory.get("player_state", {}).get("moves")
                if polls >= max_polls:
                    # Safety cap - regen or parsing isn't behaving as
                    # expected. Don't hang the run forever; stand up with
                    # whatever movement we have and let exploration resume
                    # (it will just trigger another rest cycle soon if
                    # movement is still too low, rather than stalling here).
                    self._cmd("stand")
                    break
                if moves is None or moves < min_moves:
                    time.sleep(poll_interval)
        self.save()

    def _register_block(self, block, cur_id, direction):
        """Shared by step() and bootstrap_position(): given a parsed room
        block and the room/direction we arrived from (None/None for a
        context-free bootstrap look), localize it and update the graph.
        Returns {"outcome": "moved", "room_id": ..., "is_new": ...}.

        Post-amendment (see identify_room()'s docstring): there is no more
        "tentative" outcome here. Graph-consistency-gated merging means the
        agent always either confirms an existing node (via an already-
        walked edge or an immediate-backtrack match) or creates a new one -
        it never blind-merges on text alone, so there is nothing left to
        leave unresolved.

        Also tracks `memory["last_move"]` = {"direction", "from_room"} -
        the direction just walked and the room walked from - across calls,
        purely so the *next* call can supply identify_room()'s Amendment 2
        (immediate-backtrack) context. Only updated for real moves
        (direction is not None); a context-free bootstrap look leaves it
        untouched, so it can't be mistaken for a real preceding step."""
        prev_move = self.memory.get("last_move")
        room_id, is_new, same_text_matches = identify_room(
            self.memory["rooms"], block["title"], block["exit_letters"],
            cur_id, direction,
            prev_move_dir=prev_move["direction"] if prev_move else None,
            two_ago_room_id=prev_move["from_room"] if prev_move else None,
        )

        rooms = self.memory["rooms"]
        if is_new:
            room_id = self._new_room_id()
            rooms[room_id] = {
                "title": block["title"],
                "exit_letters": block["exit_letters"],
                "exits": {},
                "blocked_exits": [],
                "desc_lines": block["desc_lines"],
                "mob_lines": block["mob_lines"],
                "sightings": 1,
                "first_seen_step": self.memory.get("step_count", 0),
                "exits_cross_check": {},
                "examined": {},
                "mob_considers": {},
                "shop": None,
                # DFS-with-backtracking bookkeeping (architecture_v2.md
                # Frontier explorer Amendment 2). untried_exits starts as
                # every exit the room reports; the explorer pops from it as
                # each direction gets tried (moved, blocked, or deferred).
                # dfs_parent/pocket_id are left None here and only set by
                # the frontier explorer: None marks "not yet incorporated
                # into any pocket's DFS tree" (true for a brand-new pocket
                # root, and true for a room that was only peeked through a
                # deferred door and hasn't been formally crossed into yet).
                "untried_exits": list(block["exit_letters"]),
                "dfs_parent": None,
                "pocket_id": None,
                "deferred_doors": [],
            }
            if same_text_matches:
                # Over-split by design: this new node looks textually
                # identical to already-known room(s), but the specific edge
                # that would confirm they're the same has not been walked.
                # Logged for visibility (this is exactly the situation
                # acceptance check 4 cares about - the agent must not
                # silently merge these), not merged.
                self.memory.setdefault("possible_duplicate_events", []).append({
                    "at_step": self.memory.get("step_count", 0),
                    "new_room_id": room_id,
                    "same_text_as": same_text_matches,
                    "title": block["title"],
                    "exit_letters": block["exit_letters"],
                })
        else:
            rooms[room_id]["sightings"] = rooms[room_id].get("sightings", 0) + 1

        # Record the edge we just walked. Deliberately one-directional: we
        # only know this direction leads there because we just walked it -
        # we do NOT assume the reverse exit takes us back (some MUD exits
        # are one-way; asserting symmetry we have not observed would be
        # exactly the kind of unearned world-knowledge this agent is built
        # to avoid). The reverse edge gets recorded on its own the first
        # time the agent actually walks it.
        if direction and cur_id is not None and cur_id in rooms:
            rooms[cur_id]["exits"][direction] = room_id

        self.memory["current_room"] = room_id
        if self.memory.get("start_room") is None:
            self.memory["start_room"] = room_id

        if direction:
            self.memory["last_move"] = {"direction": direction, "from_room": cur_id}

        return {"outcome": "moved", "room_id": room_id, "is_new": is_new}

    def step(self, direction):
        """The one funnel all movement goes through: send a direction,
        parse the reply, localize it, update the graph. Returns a dict with
        at least {"outcome": "moved"|"blocked"|"dark"}.

        Live finding (2026-07-17, DFS frontier explorer's first clean-pocket
        run): a genuine "you can't go that way" failure and a successful
        move into a room too dark to print an Exits: line both make
        parse_room_block() return None - indistinguishable without looking
        at the raw text. Treating both as "blocked" corrupts blocked_exits
        with directions that actually work fine, which is fatal for the DFS
        explorer's backtracking (it depends on retracing the exact edge it
        arrived on) in a way it never was for the old frontier-BFS policy
        (which just tried a different candidate elsewhere instead of
        getting stuck retrying the same doomed move forever). Fix: check
        for the dark-room marker specifically and grope for a lit
        neighbor - same "movement still works blind" strategy already used
        by bootstrap_position() at startup - rather than recording a false
        block.

        Also the reactive half of the rest-cycle addendum: "You are too
        exhausted" is a genuine block()-shaped reply (no Exits: line), but
        it means movement ran out mid-move, not that the exit is bad - rest,
        then retry this exact same direction rather than poisoning
        blocked_exits or giving up on real territory."""
        lines = self._cmd(direction)
        block = parse_room_block(lines)
        cur_id = self.memory["current_room"]

        if block is None:
            raw_text = " ".join(stripped for raw, stripped in lines)
            low = raw_text.lower()
            if EXHAUSTED_TEXT in low:
                self.rest_cycle()
                return self.step(direction)
            if DARK_TEXT in low:
                block = self._grope_for_light()
                if block is None:
                    # Still can't confirm anything - leave blocked_exits and
                    # current_room untouched (this was never established as
                    # a genuine block), let the caller retry or move on.
                    return {"outcome": "dark"}
                return self._register_block(block, cur_id, direction)

            # Closed door, "you can't go that way", combat interrupt, etc -
            # a genuine block. Real player behavior: note it and don't
            # retry that exact exit from here - do not move the recorded
            # current_room, do not crash.
            if cur_id is not None:
                cur = self.memory["rooms"][cur_id]
                blocked = cur.setdefault("blocked_exits", [])
                if direction not in blocked:
                    blocked.append(direction)
            return {"outcome": "blocked"}

        return self._register_block(block, cur_id, direction)

    def _grope_for_light(self):
        """Try each compass direction blind until one returns a real,
        parseable room block, or None if every direction stayed dark.
        Shared by step() (mid-exploration darkness) and bootstrap_position()
        (startup darkness) - movement still works blind in a pitch-black
        room, only the room-block text is suppressed."""
        for direction in DIRS:
            block = parse_room_block(self._cmd(direction))
            if block is not None:
                return block
        return None

    def enrich_new_room(self, room_id):
        """Maximize MUD command use on a newly-discovered room (user
        decision, 2026-07-17 - see architecture doc Components #4): beyond
        the implicit look from the room block, also run `exits`, `examine`
        on notable description nouns, `consider` on each sighted mob, and
        `list` if a mob was sighted. Pure live-text capture, no AI, no new
        world-knowledge source - only runs once per room, the first time it
        is discovered, matching "on every newly-discovered room" in the
        refinement."""
        room = self.memory["rooms"][room_id]

        # 1. exits - cross-check against the parsed Exits: line, and can
        # surface hidden/closed-door exits the passive parse missed.
        lines = self._cmd("exits")
        cross_check = {}
        for raw, stripped in lines:
            m = re.match(r"^(north|south|east|west|up|down)\s*-\s*(.+)$",
                         stripped, re.IGNORECASE)
            if m:
                letter = DIR_WORD_TO_LETTER[m.group(1).lower()]
                cross_check[letter] = m.group(2).strip()
        room["exits_cross_check"] = cross_check

        # 2. examine each notable object named in the description.
        for noun in extract_notable_nouns(room["desc_lines"]):
            lines = self._cmd(f"examine {noun}")
            text = " ".join(stripped for raw, stripped in lines)
            if EXAMINE_FAIL_TEXT in text.lower() or not text.strip():
                continue
            room["examined"][noun] = text

        # 3. consider each sighted mob (relative-difficulty read only - the
        # agent never has levels/vnums, just the game's own comparative
        # text; future combat primitives read this, not any static data).
        for mob_line in room["mob_lines"]:
            for candidate in extract_target_candidates(mob_line):
                lines = self._cmd(f"consider {candidate}")
                text = " ".join(stripped for raw, stripped in lines)
                if CONSIDER_FAIL_TEXT in text.lower() or not text.strip():
                    continue
                room["mob_considers"][mob_line] = {
                    "guessed_keyword": candidate,
                    "result": text,
                }
                break

        # 4. list, only in rooms where a mob was sighted (matches real-player
        # behavior; avoids spamming `list` in every unknown room).
        if room["mob_lines"]:
            lines = self._cmd("list")
            text_lines = [stripped for raw, stripped in lines]
            text = "\n".join(text_lines)
            if SHOP_FAIL_TEXT not in text.lower():
                room["shop"] = {"transcript": text, "lines": text_lines}

        self.save()

    def bootstrap_position(self):
        """Always re-check identity on startup (fresh process = an
        'unexpected large-scale discontinuity' per the architecture doc's
        Localizer section - never assume where we are, even on resume).

        A plain `look` can come back with no room block at all if it is
        pitch black (confirmed live: outdoor tbaMUD rooms go dark at night,
        showing only "It is pitch black..." with no title/Exits: line - a
        transient day/night state, not a permanent room property). Movement
        still works blind in that state, so instead of crashing (the LLM
        boundary rule says the engine must never crash on a live-text
        surprise - the same discipline applies here), grope for a lit
        neighbor by trying each direction until one returns a real room
        block, same as a real player fumbling in the dark would."""
        block = parse_room_block(self._cmd("look"))
        if block is None:
            block = self._grope_for_light()
        if block is None:
            # Still nothing after trying every direction blind - genuinely
            # stuck (e.g. a solitary dark dead end). Fall back to whatever
            # position memory already has, rather than crashing; the
            # frontier explorer will find out soon enough if that is stale.
            print("[bootstrap] no room block after blind probing in the dark; "
                  "keeping prior current_room from memory (best effort).")
            return self.memory.get("current_room")

        result = self._register_block(block, None, None)
        room_id = result["room_id"]
        self.save()
        if result["is_new"]:
            self.enrich_new_room(room_id)

        # First-ever room of a fresh memory becomes pocket 0's root (DFS
        # frontier explorer, architecture_v2.md Amendment 2). A resumed run
        # already has a non-empty pocket_stack, so this only fires once per
        # fresh agent_memory.json.
        if not self.memory.get("pocket_stack"):
            pocket_id = str(self.memory.get("next_pocket_id", 0))
            self.memory["next_pocket_id"] = int(pocket_id) + 1
            pocket = {"deferred": [], "room_count": 0, "profile": {}}
            self.memory.setdefault("pockets", {})[pocket_id] = pocket
            self.memory["pocket_stack"] = [pocket_id]
            room = self.memory["rooms"][room_id]
            room["dfs_parent"] = None
            room["pocket_id"] = pocket_id
            room.setdefault("untried_exits", list(room["exit_letters"]))
            update_pocket_profile(pocket, room["desc_lines"])
            self.save()

        return room_id


# ---------------------------------------------------------------------------
# Pathfinder (architecture doc, Components #6) - plain BFS over the agent's
# own recorded exits, hop by hop, verifying identity at each step.
# ---------------------------------------------------------------------------

def bfs_path(rooms, start_id, goal_pred):
    """BFS over rooms[*]['exits'] from start_id to the nearest room
    satisfying goal_pred. Returns a list of directions, or None if
    unreachable from what the agent currently knows."""
    if start_id is None or start_id not in rooms:
        return None
    if goal_pred(start_id):
        return []
    visited = {start_id}
    queue = deque([(start_id, [])])
    while queue:
        rid, path = queue.popleft()
        for d, nxt in rooms[rid]["exits"].items():
            if nxt is None or nxt not in rooms or nxt in visited:
                continue
            new_path = path + [d]
            if goal_pred(nxt):
                return new_path
            visited.add(nxt)
            queue.append((nxt, new_path))
    return None


def navigate_to(agent, target_room_id):
    """Walk the agent from its current room to target_room_id via BFS over
    its own graph, verifying each hop's resulting room against the expected
    next node before continuing; on mismatch, stop rather than push further
    blind (architecture doc, Components #6)."""
    rooms = agent.memory["rooms"]
    path = bfs_path(rooms, agent.memory["current_room"], lambda rid: rid == target_room_id)
    if path is None:
        return False
    for direction in path:
        expected_next = rooms[agent.memory["current_room"]]["exits"].get(direction)
        result = agent.step(direction)
        if result["outcome"] != "moved" or result["room_id"] != expected_next:
            return False
    return agent.memory["current_room"] == target_room_id


# ---------------------------------------------------------------------------
# Frontier explorer (architecture doc, Components #3)
#
# Amendment 2 (2026-07-17) replaced the original nearest-unvisited-frontier
# -first policy (with its later novelty-penalty patch) ENTIRELY. Root cause
# of the milestone-1 blocker was never room identity (the Localizer above is
# conclusively correct) - it was that nearest-first happily wanders far from
# spawn, live-reproduced wandering through a city gate into a disconnected,
# hostile, one-way dead end (a guarded tower entrance) before ever
# exhausting the safe pocket around spawn where a bakery would plausibly be.
#
# Replacement: DFS with backtracking + per-room untried-exit bookkeeping,
# plus pocket-boundary deferral. See next_dfs_action()'s docstring for the
# decision policy and is_pocket_boundary()'s for the deferral heuristic.
# ---------------------------------------------------------------------------

# Heuristic "theme vocabulary" extraction for the pocket-boundary check:
# words of 4+ letters, minus the same generic stopword list already used
# for noun/keyword extraction elsewhere in this file. Deliberately coarse -
# same accepted-imprecision spirit as everywhere else this file guesses at
# meaning from live text (bakery keywords, examine-target nouns).
THEME_WORD_RE = re.compile(r"[a-z]{4,}")

# Empirical check for real zone-transition signage (architecture doc,
# Frontier explorer Amendment 2, "check for real signage first, cheaply"):
# live probes against the running tbaMUD 2025 container (help files +
# several room transitions, dev-session 2026-07-17) found no such banner
# text anywhere - tbaMUD's stock room-block output is just title/desc/exits,
# with no automatic "you are now entering..." line. This regex is kept as a
# cheap, layered-on-top upgrade per the doc (never a dependency) in case a
# specific zone happens to have one; matches are logged to
# memory["zone_signage_events"] and treated as a strong, forced boundary
# signal, but the heuristic in is_pocket_boundary() is what actually carries
# milestone 1 since no live match has ever been observed.
ZONE_SIGNAGE_RE = re.compile(
    r"you (?:are now entering|have entered)|welcome to\b|entering the\b",
    re.IGNORECASE,
)


def extract_theme_words(desc_lines):
    text = " ".join(desc_lines).lower()
    return [w for w in THEME_WORD_RE.findall(text) if w not in STOPWORDS]


def detect_zone_signage(desc_lines):
    """Returns the matched signage phrase, or None. See ZONE_SIGNAGE_RE."""
    m = ZONE_SIGNAGE_RE.search(" ".join(desc_lines))
    return m.group(0) if m else None


def update_pocket_profile(pocket, desc_lines):
    """Fold one room's description into the pocket's rolling vocabulary
    profile (word -> sighting count) and bump its room_count. Called once
    per room formally incorporated into a pocket (not for rooms that are
    only peeked through a deferred door and stepped back from)."""
    profile = pocket.setdefault("profile", {})
    for w in extract_theme_words(desc_lines):
        profile[w] = profile.get(w, 0) + 1
    pocket["room_count"] = pocket.get("room_count", 0) + 1


def is_pocket_boundary(pocket, new_desc_lines, min_rooms=2, min_new_words=4,
                        overlap_threshold=0.2):
    """Fuzzy pocket-boundary heuristic (architecture_v2.md Frontier explorer
    Amendment 2): a newly-discovered room "looks like a different area" from
    the current pocket if its description shares little vocabulary with the
    pocket's rolling word profile so far. Necessarily fuzzy - no
    ground-truth sector/zone data exists to check a guess against (same
    spirit as the bakery keyword family elsewhere in this file).

    Requires at least `min_rooms` rooms already folded into the pocket's
    profile and at least `min_new_words` distinct significant words in the
    new room's own description before it will ever call a boundary - too
    little signal in either direction is treated as "can't tell", not as a
    boundary, so a pocket's own first room or a terse description never
    triggers a spurious deferral. Otherwise: boundary if fewer than
    `overlap_threshold` of the new room's distinct theme words have been
    seen anywhere in the pocket so far."""
    if pocket.get("room_count", 0) < min_rooms:
        return False
    new_words = set(extract_theme_words(new_desc_lines))
    if len(new_words) < min_new_words:
        return False
    profile = pocket.get("profile", {})
    overlap = sum(1 for w in new_words if profile.get(w, 0) > 0)
    return (overlap / len(new_words)) < overlap_threshold


def score_deferred_door(rooms, door, family):
    """Pure: how relevant does this deferred door's already-observed
    destination room look for the current challenge's keyword family?
    `door` is a [origin_room_id, direction] pair; the destination room was
    already discovered (and enriched) at the moment the boundary heuristic
    deferred it, so its title/desc/mob text is sitting right there in
    `rooms` - reuses `_keyword_hit()`, the exact same fuzzy matching
    find_place already uses to recognize a bakery match, rather than
    inventing a second heuristic (or a third LLM call - explicitly
    considered and rejected, see architecture_v2.md: it would break the
    "exactly two LLM calls" boundary this project documents for no real
    benefit, since the data needed is already sitting in memory). Returns
    the keyword-hit count (0 if the destination isn't known for some reason,
    or matches nothing)."""
    origin_id, direction = door
    origin = rooms.get(origin_id)
    if not origin:
        return 0
    target_id = origin.get("exits", {}).get(direction)
    target = rooms.get(target_id) if target_id else None
    if not target:
        return 0
    return len(_keyword_hit(target, family))


def choose_deferred_door(rooms, deferred, family):
    """Pure: which deferred door to cross next, when a pocket is fully
    exhausted and one or more doors are queued (architecture_v2.md
    addendum, 2026-07-17). Score each door's already-observed destination
    against the keyword family and cross the highest-scoring one first.
    Ties, and the case where nothing scores above zero (no `family` given,
    or a challenge type find_place's keyword matching doesn't apply to),
    fall back to oldest-deferred-first (list order) - the original FIFO
    behavior, so this never behaves worse than before when there is no
    signal to prefer one door over another."""
    if not deferred:
        return None
    if not family:
        return deferred[0]
    best = deferred[0]
    best_score = score_deferred_door(rooms, best, family)
    for door in deferred[1:]:
        score = score_deferred_door(rooms, door, family)
        if score > best_score:
            best, best_score = door, score
    return best


def next_dfs_action(memory, keyword_family=None):
    """Pure decision function: given the agent's full memory state, decides
    what the frontier explorer should try next WITHOUT touching the
    network. Priority order (this IS the DFS-with-backtracking + pocket-
    deferral policy from architecture_v2.md's Frontier explorer Amendment
    2):

      1. An untried exit of the CURRENT room, if any - always preferred
         over anything else, so the agent never jumps to a distant frontier
         room while there is still unfinished business right where it is
         standing.
      2. If none, and the current room has a dfs_parent (it was reached by
         descending from another room in this pocket), backtrack there -
         walk the reverse of however it was reached.
      3. If none, and the current room IS a pocket root (dfs_parent is
         None) - by the point we reach here the whole pocket's DFS tree is
         provably exhausted (each ancestor only ever backtracks past itself
         once its own untried_exits are empty, so reaching an exhausted
         root means nothing anywhere in the pocket is left untried). If the
         pocket has deferred "door to elsewhere" exits queued, cross the
         one whose already-observed destination scores highest against
         `keyword_family` (see choose_deferred_door()) - falls back to
         oldest-deferred-first when nothing scores above zero.
      4. If the pocket has no untried exits AND no deferred doors left, it
         is fully done - pop it off the pocket stack. The caller re-invokes
         this function afterward against whatever pocket is now on top.
      5. If the pocket stack is empty, there is nothing left anywhere that
         this agent's own graph can reach - done.

    Returns one of:
      {"kind": "descend",   "direction": d}
      {"kind": "backtrack", "direction": d}
      {"kind": "cross_door","room_id": r, "direction": d}
      {"kind": "pop_pocket"}
      {"kind": "done"}
    """
    rooms = memory["rooms"]
    cur_id = memory["current_room"]
    cur = rooms[cur_id]

    untried = [d for d in cur.get("untried_exits", [])
               if d not in cur.get("blocked_exits", [])]
    if untried:
        return {"kind": "descend", "direction": untried[0]}

    parent = cur.get("dfs_parent")
    if parent is not None:
        return {"kind": "backtrack", "direction": OPPOSITE_DIR[parent["dir"]]}

    pocket_stack = memory.get("pocket_stack", [])
    if not pocket_stack:
        return {"kind": "done"}

    pocket = memory["pockets"][pocket_stack[-1]]
    if pocket.get("deferred"):
        door_room_id, door_dir = choose_deferred_door(rooms, pocket["deferred"], keyword_family)
        return {"kind": "cross_door", "room_id": door_room_id, "direction": door_dir}

    return {"kind": "pop_pocket"}


def explore_one_step(agent, keyword_family=None):
    """Execute one decision from next_dfs_action() against the live MUD and
    fold the observation back into memory. Returns
    (progressed, newly_discovered_room_id) - same contract as before this
    amendment: progressed=False means nothing reachable is left to try
    anywhere in the agent's own graph.

    `keyword_family` is threaded straight through to next_dfs_action() so
    that if a pocket is exhausted with multiple deferred doors queued, the
    one whose already-observed destination looks most relevant gets crossed
    first (see choose_deferred_door()) - optional, defaults to plain
    oldest-deferred-first when not exploring toward a keyword goal.

    Proactive half of the rest-cycle addendum (architecture_v2.md, 2026-07-17):
    checked before every decision, using whatever player_state the last
    command already told us, so the explorer doesn't spend a doomed move
    attempt on 0 movement when it can rest first instead. step()'s
    EXHAUSTED_TEXT branch is the reactive fallback for a move that fails on
    cost despite the last-seen prompt still showing a small positive
    number."""
    memory = agent.memory
    rooms = memory["rooms"]
    if needs_rest(memory.get("player_state")):
        agent.rest_cycle()
    action = next_dfs_action(memory, keyword_family=keyword_family)

    if action["kind"] == "done":
        return False, None

    if action["kind"] == "pop_pocket":
        memory["pocket_stack"].pop()
        return (bool(memory["pocket_stack"]), None)

    if action["kind"] == "backtrack":
        # Live finding (2026-07-19, first genuinely clean run): "walk the
        # compass-opposite of however I arrived" is NOT reliably the way
        # back to the parent - it only IS the parent if that exact reverse
        # direction has never been separately claimed by one of this room's
        # OWN other (forward) exits. A room's four compass exits are
        # independent facts about the MUD's geometry, not a guaranteed
        # reciprocal grid; the direction opposite of "how I arrived" can
        # easily already be a genuine, different, unrelated connection
        # discovered earlier in this room's own DFS descent (confirmed
        # live: a "Great Field" room entered via north had its own south
        # exit already recorded as leading to a completely different child
        # room from an earlier descend - "backtracking" via south
        # deterministically walked into that child instead of the parent,
        # every time, forever). Verify the landing room; if it isn't the
        # expected parent, retrying the same move would oscillate between
        # these two rooms indefinitely (exactly what was live-observed:
        # current_room frozen on one "Great Field" node for 50+ steps).
        from_room_id = memory["current_room"]
        expected_parent_id = rooms[from_room_id]["dfs_parent"]["room"]
        agent.step(action["direction"])
        if memory["current_room"] != expected_parent_id:
            # No known way back to the true parent from here. Sever the
            # link rather than retry a doomed move - this room now behaves
            # like an exhausted pocket root on the next call (its own
            # deferred doors, if any, get a chance; otherwise the pocket
            # gets popped), which self-resolves once ordinary backtracking
            # returns here from wherever this move actually landed. The
            # parent's own still-untried exits become unreachable from this
            # branch - an accepted limitation, same spirit as the
            # already-documented "some exits are genuinely one-way" case -
            # logged for visibility, not silently dropped.
            rooms[from_room_id]["dfs_parent"] = None
            memory.setdefault("unreachable_parent_events", []).append({
                "at_step": memory.get("step_count", 0),
                "room_id": from_room_id,
                "attempted_parent": expected_parent_id,
                "attempted_direction": action["direction"],
                "landed_at": memory["current_room"],
            })
        return True, None

    if action["kind"] == "cross_door":
        door_room_id, direction = action["room_id"], action["direction"]
        if memory["current_room"] != door_room_id:
            if not navigate_to(agent, door_room_id):
                # Can't currently reach the door room (shouldn't normally
                # happen - it's a recorded edge). Leave it queued; the next
                # call will just retry rather than dropping it silently.
                return True, None
        cur = rooms[door_room_id]
        result = agent.step(direction)
        if result["outcome"] != "moved":
            # Door no longer passable - drop it and move on rather than
            # retrying forever.
            cur["deferred_doors"] = [d for d in cur.get("deferred_doors", [])
                                      if d != direction]
            pocket = memory["pockets"][memory["pocket_stack"][-1]]
            pocket["deferred"] = [e for e in pocket["deferred"]
                                   if e != [door_room_id, direction]]
            return True, None

        pocket = memory["pockets"][memory["pocket_stack"][-1]]
        pocket["deferred"] = [e for e in pocket["deferred"]
                               if e != [door_room_id, direction]]

        new_room_id = result["room_id"]
        new_room = rooms[new_room_id]
        discovered_id = None
        if result["is_new"]:
            # Defensive fallback only - normally the peek that originally
            # deferred this door already created and enriched this node, so
            # the edge resolves to it (is_new=False) and this branch never
            # fires. Handled anyway so a desync can't crash the run.
            new_room.setdefault("untried_exits", list(new_room["exit_letters"]))
            agent.enrich_new_room(new_room_id)
            discovered_id = new_room_id

        new_pocket_id = str(memory.get("next_pocket_id", 0))
        memory["next_pocket_id"] = int(new_pocket_id) + 1
        new_pocket = {"deferred": [], "room_count": 0, "profile": {}}
        memory["pockets"][new_pocket_id] = new_pocket
        memory["pocket_stack"].append(new_pocket_id)
        new_room["dfs_parent"] = None
        new_room["pocket_id"] = new_pocket_id
        update_pocket_profile(new_pocket, new_room.get("desc_lines", []))
        return True, discovered_id

    # action["kind"] == "descend"
    direction = action["direction"]
    from_room_id = memory["current_room"]
    cur_before = rooms[from_room_id]
    cur_before["untried_exits"] = [d for d in cur_before.get("untried_exits", [])
                                    if d != direction]

    result = agent.step(direction)
    if result["outcome"] != "moved":
        return True, None

    new_room_id = result["room_id"]
    new_room = rooms[new_room_id]

    if not result["is_new"]:
        # A known room reached via a not-previously-recorded edge (loop
        # closure). Live finding (2026-07-19): staying here is a bug, not
        # just "nothing else to do" - the room we just LEFT (from_room_id)
        # may still have its own untried exits, and if we don't return to
        # it now, next_dfs_action will look at wherever this loop landed
        # instead, potentially trying THAT room's other untried exits
        # first and permanently orphaning from_room_id's remaining ones
        # (confirmed live: a room's first-tried exit looped back to its
        # own parent, which still had untried exits of its own - the
        # child's real, still-untried second exit was never revisited).
        # Step back immediately, same "step in, step back" pattern already
        # used for pocket-boundary deferral - reliable because this is a
        # true single-hop reversal of the move just made (Amendment 2's
        # immediate-backtrack recognition, or the now-just-recorded edge
        # itself, both resolve it correctly).
        agent.step(OPPOSITE_DIR[direction])
        return True, None

    agent.enrich_new_room(new_room_id)

    pocket = memory["pockets"][memory["pocket_stack"][-1]]
    signage = detect_zone_signage(new_room.get("desc_lines", []))
    if signage:
        memory.setdefault("zone_signage_events", []).append({
            "at_step": memory.get("step_count", 0),
            "room_id": new_room_id,
            "matched": signage,
        })
    boundary = bool(signage) or is_pocket_boundary(pocket, new_room.get("desc_lines", []))

    if boundary:
        rooms[from_room_id]["deferred_doors"].append(direction)
        pocket["deferred"].append([from_room_id, direction])
        agent.step(OPPOSITE_DIR[direction])  # step back; immediate-backtrack
        # recognition resolves this to from_room_id without minting a new
        # node. new_room stays dfs_parent=None / pocket_id=None - "peeked,
        # not yet formally entered" - until this door is crossed for real.
    else:
        new_room["dfs_parent"] = {"room": from_room_id, "dir": direction}
        new_room["pocket_id"] = memory["pocket_stack"][-1]
        update_pocket_profile(pocket, new_room.get("desc_lines", []))

    return True, new_room_id


def probe_reverse_edge(agent, room_id, arrived_via):
    """Opportunistic, bounded check (at most one extra command) on first
    arrival at a new room: try the exact opposite of the direction just
    walked, to see if it leads back the way we came.

    Movement edges are otherwise recorded strictly one-directionally (see
    Agent._register_block's comment) since some tbaMUD exits are genuinely
    one-way or listed-but-non-functional (confirmed live: a room whose
    Exits: line listed n/e/w had only n actually work - "e"/"w" both
    returned "Alas, you cannot go that way..."), so asserting symmetry the
    agent has not itself observed would be exactly the kind of unearned
    world-knowledge it is built to avoid.

    But leaving every edge one-directional has a real, live-observed cost:
    the frontier explorer only ever walks recorded forward edges, so a
    room's *other*, still-unexplored exits become permanently unreachable
    if nothing links back to it - confirmed live in a 19-room exploration
    that plateaued with two known-unexplored exits BFS could no longer
    reach. This probe earns the reverse edge instead of assuming it: one
    extra, verified command per newly-discovered room, symmetric with the
    "maximize command use" policy already applied to exits/examine/
    consider/list. Whatever room this move actually lands in - the origin,
    if it is a normal two-way passage, or somewhere else entirely, if not -
    is registered through the same step() pipeline as any other move, so
    state never desyncs; the next explore_one_step() call simply re-plans
    its frontier path from wherever the agent actually ends up. If that
    landing room is itself brand new, it gets the same enrichment pass as
    any other newly-discovered room.

    DISABLED (2026-07-17, post-Localizer-amendment finding): live-tested
    together with the graph-consistency-gated identify_room() fix (see that
    function's docstring) and found to compound badly. The probe's landing
    room, by construction, has an empty exits dict the first time it is
    reached this way - so under the new "only merge on a confirmed edge"
    rule, the probe can never graph-confirm a return to the room it just
    left, even when it physically is that same room. Reproduced live: a
    repeating Main-Street-then-Weapon-Shop layout turned into an endless
    n/s oscillation, minting a brand-new duplicate node pair on every
    round trip (28 rooms discovered, 26 of them flagged
    possible_duplicate_events) instead of ever moving further down the
    street - it starved out real exploration rather than aiding it. Since
    the architecture doc's amendment explicitly calls
    probe_reverse_edge()-based retroactive corroboration a should-have, not
    required, for this milestone, the safer call is to disable the probe
    entirely rather than ship a mechanism proven to actively hurt
    exploration under the new merge rule - re-enabling it needs pairing
    with the retroactive-merge logic the doc anticipates (so a probe that
    lands somewhere textually identical to where it started gets merged
    back, not minted as a new node), which is future work, not this pass.
    Flagged to the Orchestrator rather than silently redesigning further."""
    return


# ---------------------------------------------------------------------------
# Goal executor / primitives (architecture doc, Components #2)
# ---------------------------------------------------------------------------

PRIMITIVES = {"find_place"}


def _keyword_hit(room, family):
    text = " ".join([room["title"] or ""] + room["desc_lines"] + room["mob_lines"]).lower()
    return [k for k in family if k in text]


def build_report(agent, room_id, matched_keywords):
    rooms = agent.memory["rooms"]
    room = rooms[room_id]
    surroundings = []
    for d, nxt_id in sorted(room["exits"].items()):
        nxt = rooms.get(nxt_id)
        if not nxt:
            continue
        surroundings.append({
            "direction": d,
            "title": nxt["title"],
            "short_desc": " ".join(nxt["desc_lines"])[:200],
            "notable": sorted(nxt.get("examined", {}).keys()),
        })
    return {
        "status": "found",
        "room_id": room_id,
        "title": room["title"],
        "matched_keywords": matched_keywords,
        "goods": room["shop"]["transcript"] if room.get("shop") else None,
        "surroundings": surroundings,
        "mob_considers": room.get("mob_considers", {}),
        "rooms_discovered": len(rooms),
    }


# How often (in exploration steps) to opportunistically refresh the fuller
# `score` picture during normal exploration, independent of resting -
# architecture_v2.md addendum, 2026-07-17 ("run score periodically... into
# that same player_state"). Rest cycles already call `score` repeatedly on
# their own while polling for movement recovery, so this interval only
# matters for stretches where no rest cycle happens to fire.
PLAYER_STATE_SCORE_INTERVAL = 25


def run_find_place(agent, keyword, max_steps=2000):
    family = KEYWORD_FAMILIES.get((keyword or "").lower(), [(keyword or "").lower()])
    rooms = agent.memory["rooms"]

    # Already-known match from a prior run (e.g. resumed memory)? Navigate
    # there and report instead of re-exploring from scratch.
    for rid, r in rooms.items():
        hit = _keyword_hit(r, family)
        if hit:
            if agent.memory["current_room"] != rid:
                navigate_to(agent, rid)
            return build_report(agent, rid, hit)

    steps = agent.memory.get("step_count", 0)
    while steps < max_steps:
        progressed, discovered_id = explore_one_step(agent, keyword_family=family)
        steps += 1
        agent.memory["step_count"] = steps
        if steps % PLAYER_STATE_SCORE_INTERVAL == 0:
            agent._cmd("score")
        agent.save()
        if discovered_id is not None:
            hit = _keyword_hit(rooms[discovered_id], family)
            if hit:
                return build_report(agent, discovered_id, hit)
        if not progressed:
            break

    return {
        "status": "not_found",
        "steps": steps,
        "rooms_discovered": len(rooms),
        "possible_duplicate_events": len(agent.memory.get("possible_duplicate_events", [])),
    }


def run_primitive(agent, primitive, params, max_steps=2000):
    if primitive not in PRIMITIVES:
        raise ValueError(f"primitive not implemented: {primitive}")
    if primitive == "find_place":
        return run_find_place(agent, params.get("keyword", ""), max_steps=max_steps)
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# Memory persistence
# ---------------------------------------------------------------------------

def load_memory():
    if MEMORY_PATH.exists():
        try:
            with open(MEMORY_PATH) as fh:
                data = json.load(fh)
            print(f"[memory] resumed: {len(data.get('rooms', {}))} rooms known, "
                  f"current_room={data.get('current_room')}")
            return data
        except (OSError, json.JSONDecodeError) as e:
            print(f"[memory] failed to load {MEMORY_PATH}: {e} - starting fresh")
    return {
        "next_id": 1,
        "rooms": {},
        "current_room": None,
        "start_room": None,
        "step_count": 0,
        "possible_duplicate_events": [],
        "last_move": None,
        # Continuous player-state tracking (architecture_v2.md addendum,
        # 2026-07-17): hp/mana/moves parsed from every prompt, plus the
        # fuller score fields (level/gold/exp/armor_class/alignment/...)
        # parsed whenever `score` runs (rest cycles, and every
        # PLAYER_STATE_SCORE_INTERVAL steps during normal exploration).
        "player_state": {},
        # DFS-with-backtracking frontier explorer state (architecture_v2.md
        # Amendment 2). pocket_stack[-1] is the currently-active pocket;
        # pockets maps pocket_id (str) -> {"deferred": [[room_id, dir], ...],
        # "room_count": int, "profile": {word: count}}. Initialized lazily
        # by Agent.bootstrap_position() on the first room of a fresh run.
        "pocket_stack": [],
        "pockets": {},
        "next_pocket_id": 0,
        "zone_signage_events": [],
        # Logged whenever a dfs_parent backtrack's compass-reverse guess
        # didn't land at the expected parent room (architecture_v2.md
        # Frontier explorer, 2026-07-19 live finding) - see the "backtrack"
        # branch of explore_one_step() for why this can happen and why it's
        # an accepted limitation, not a crash.
        "unreachable_parent_events": [],
    }


# ---------------------------------------------------------------------------
# CLI entry point. Orchestration only - the two LLM calls it invokes live
# entirely inside challenge_front_door.py (imported lazily below so that
# `import challenge_agent` never requires the anthropic package or an API
# key unless main() actually runs).
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Challenge agent for the tbaMUD test environment.")
    parser.add_argument("challenge", nargs="?", default=None,
                         help="Plain-language challenge, e.g. 'find the bakery'")
    parser.add_argument("--username", default=os.environ.get("MUD_USERNAME"),
                         required="MUD_USERNAME" not in os.environ,
                         help="MUD character name (or set MUD_USERNAME)")
    parser.add_argument("--password", default=os.environ.get("MUD_PASSWORD"),
                         required="MUD_PASSWORD" not in os.environ,
                         help="MUD character password (or set MUD_PASSWORD) - never hardcode this")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--no-phrase", action="store_true",
                         help="skip the optional LLM phrasing call, print the raw result only")
    parser.add_argument("--skip-front-door", metavar="PRIMITIVE",
                         help="dev/testing only: bypass the LLM front door and call this "
                              "primitive directly (e.g. --skip-front-door find_place "
                              "--keyword bakery). Never used for the graded run path.")
    parser.add_argument("--keyword", default=None,
                         help="param for --skip-front-door find_place")
    args = parser.parse_args()

    if args.skip_front_door:
        primitive = args.skip_front_door
        params = {"keyword": args.keyword} if args.keyword else {}
    else:
        if not args.challenge:
            parser.error("a challenge is required unless --skip-front-door is used")
        import challenge_front_door as front_door
        interpretation = front_door.interpret_challenge(args.challenge)
        if not interpretation.get("supported"):
            print(f"unsupported: {interpretation.get('reason')}")
            return
        primitive = interpretation["primitive"]
        params = interpretation.get("params", {})

    if primitive not in PRIMITIVES:
        print(f"primitive not implemented: {primitive}")
        return

    memory = load_memory()
    client = MudClient(args.username, args.password)
    client.connect()
    client.login()
    agent = Agent(client, memory)
    agent.bootstrap_position()

    try:
        result = run_primitive(agent, primitive, params, max_steps=args.max_steps)
    finally:
        agent.save()
        client.close()

    print(json.dumps(result, indent=2, default=str))

    if not args.no_phrase and not args.skip_front_door:
        try:
            import challenge_front_door as front_door
            print("\n" + front_door.phrase_report(result))
        except Exception as e:
            print(f"\n[LLM phrasing unavailable, showing raw result only: {e}]")


if __name__ == "__main__":
    main()
