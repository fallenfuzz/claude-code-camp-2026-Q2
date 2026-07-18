# Challenge Agent

An autonomous player agent for the tbaMUD test environment (the same
CircleMUD-derived server `journey_map/` observes). Give it a plain-language
challenge - milestone 1 supports "find the bakery" - and it explores the
live game world, builds its own memory purely from what it observes through
play, navigates to the target, and reports what it found.

Full design rationale (world-model rules, localizer design, keyword
heuristics) lives in this project's `handoffs/architecture_v2.md`
(workspace-side, not part of this repo).

## Is this an agent, or just a script? - the LLM boundary

There are exactly **two** LLM call sites in this entire codebase, both in
`challenge_front_door.py` and nowhere else:

1. `interpret_challenge()` - turns the plain-language challenge text into a
   known execution-engine primitive call (or `unsupported` with a reason).
   The prompt contains zero world knowledge - no map, no mob data, nothing
   about this specific game.
2. `phrase_report()` (optional) - turns the engine's raw structured result
   into a short natural-language sentence. If this call fails or is
   skipped, the caller falls back to printing the raw structured result
   directly - this step is never load-bearing for correctness.

Everything else - exploration, room identity/localization, shop and mob
detection, pathfinding, the report itself - is deterministic Python in
`challenge_agent.py`, which never imports `anthropic` and never sees an API
key. A reviewer can verify this in one command:

```
grep -rn "anthropic" *.py   # only challenge_front_door.py should match
```

Both LLM calls use Haiku 4.5 (`claude-haiku-4-5-20251001`) - the job (map
text to a known primitive; phrase a short report) is genuinely thin and
does not need a bigger model.

## What it does NOT do

- It never opens any file under `../preview/data/world/` (the parsed
  static world data journey_map legitimately uses as ground truth for its
  own, different job - faithful cartography). This agent is deliberately
  self-taught instead: it only knows what it has personally observed via
  `look`, movement, `exits`, `examine`, `consider`, and `list`.
- It never assumes it knows where it is. Every run re-checks its current
  room from a live `look` before doing anything else, and re-derives room
  identity from scratch on every single observation (title + exit letters
  + movement history) rather than trusting persisted state.

## How it explores

On every **newly-discovered** room (not on repeat visits), the agent
maximizes what it learns before moving on, the way a curious real player
would:

- the room block itself (title, description, exits, any mob presence) from
  the implicit look that comes with every successful move
- `exits` - cross-checks the parsed exits line and can surface exits the
  passive parse missed
- `examine <noun>` for notable objects guessed from the room description
  (words following "a"/"an"/"some" - the same kind of extra-desc keyword a
  real player would try)
- `consider <mob>` for each mob sighted, guessing at its keyword the same
  way (the agent never has vnums or stats - only the room-presence text it
  can see, same as a player)
- `list`, but only in rooms where a mob was actually sighted (a real player
  does not try to shop in an empty room)

All of this is captured verbatim into `agent_memory.json` (gitignored,
per-run state - same treatment as journey_map's `journey_state.json`).
Killing and restarting the process resumes from this file instead of
starting cold.

The frontier explorer always walks to the *nearest* room with an unexplored
exit (BFS over its own discovered graph), never a scripted replay of a
precomputed shortest path - it has no idea in advance how big the world is.

Movement edges are recorded strictly one-directionally: walking `n` from
room A to room B only ever records `A --n--> B`, never an assumed reverse
`B --s--> A` (some exits are genuinely one-way, and confirmed live: one
room's `Exits:` line listed `n e w` but only `n` actually worked - `e`/`w`
both returned "Alas, you cannot go that way..."). `probe_reverse_edge()`
exists to earn that reverse edge instead of assuming it, but is currently
**disabled** (see "Resolved limitations" below - superseded by immediate-
backtrack recognition, which handles the common case it was meant for
without the failure mode it introduced).

### Localizer amendments (2026-07-17)

Three rounds of live testing against the real MUD produced three amendments
to the original spec, in this order:

1. **Graph-consistency-gated matching.** A room is only ever treated as
   "the same room already known" if the specific edge `(last_room,
   direction)` is already recorded in memory and points at a room whose
   signature matches what was just observed. If that exact edge has never
   been walked before, the observation is a brand-new node - even if it
   looks textually identical (same title, same exit letters) to one or
   more already-known rooms. This replaced an earlier "exactly one text
   match -> merge" rule that had a real bug: a run of 3+ consecutive rooms
   sharing a title and exit letters (e.g. a long street built from
   repeating segments) silently collapsed into the first-seen room, since
   the first repeat was always the *only* known text match at that moment.
   Unit-tested (`test_repeated_identical_rooms_do_not_silently_merge`) and
   confirmed live: a run correctly created 14 distinct "Main Street" nodes
   and 14 distinct "The Weapon Shop" nodes instead of collapsing them.
2. **Novelty-penalized frontier scoring.** The graph-consistency fix trades
   false-merge risk for over-splitting risk (logged in
   `possible_duplicate_events`), which on its own let the frontier explorer
   thrash: several freshly-over-split nodes are all cheaply "nearest" to
   each other, so plain nearest-first got stuck oscillating in that cluster
   instead of pushing onward. Fix: `find_frontier_path()` deprioritizes any
   frontier room whose own `(title, exit_signature)` already matches 2+
   known rooms, using nearest-distance only as the tiebreak among
   equally-novel candidates.
3. **Immediate-backtrack recognition.** Novelty-penalized scoring alone
   wasn't enough for a *tight two-room bidirectional pair* (e.g. street <->
   shop): the current room is always trivially "nearest to itself" (0
   hops), so the novelty comparison never got a chance to fire - live-
   reproduced as 189 rooms discovered, 187 of them phantom duplicates of
   just 2 real rooms, from a street/shop ping-pong. Fix: if the direction
   just walked is the exact geometric opposite of the direction walked on
   the immediately preceding step, and the arrival's signature matches the
   room occupied *two steps ago*, treat it as that exact room. This only
   fires for a literal one-step-there-one-step-back reversal, so a genuine
   forward chain of look-alike rooms (amendment 1's target) still gets
   distinct nodes - regression-tested explicitly
   (`test_does_not_fire_without_a_true_direction_reversal`).

**Live confirmation of #3:** resumed the 189-room thrashed run with the fix
active. Across the next 397 steps, the "Main Street"/"The Weapon Shop"
counts stayed exactly frozen (95/96 - zero new duplicates), while the agent
discovered 37 more, genuinely distinct rooms (a cave system, foothills, a
valley, the East Gate of Midgaard, and eventually a hostile dead-end zone -
see below). The ping-pong bug is conclusively fixed.

### Resolved limitations (superseded, kept here for history)

- ~~`probe_reverse_edge()` disabled~~ - superseded by immediate-backtrack
  recognition above, which solves the same problem (earning a reverse edge
  instead of assuming one) without that mechanism's failure mode (it could
  never graph-confirm its own landing room, since that room's edge history
  is always empty on first arrival).
- ~~Nearest-frontier-first gets trapped on a repeating layout~~ - solved by
  the novelty penalty (#2 above) for the general case; the specific
  tight-loop case it couldn't reach (because the only "candidate" was
  always the current room itself) is solved by #3.

### Known limitation: this world's tight-loop/hostile-zone geometry

Milestone 1's `find_place("bakery")` was not completed end-to-end this
session - not because of a localizer or frontier bug (all three amendments
above are confirmed correct, unit-tested, and live-verified), but because
undirected frontier-first exploration from this particular spawn area
wandered through a very long "Main Street"/"Weapon Shop" corridor and then
into a genuinely hostile, one-way dead end: "The Entrance To The High Tower
of Sorcery," guarded by a diamond golem and shadow guardians, with a closed
gate (`(n)`, tbaMUD's closed-door notation - the agent does not attempt
`open`, another accepted milestone-1 scope gap alongside no stamina/rest
management) and a narrative one-way entrance ("the trail back south has
disappeared behind you"). A real player without combat capability would
face the same dead end. No damage was taken (confirmed via `score` - full
HP) and the character was left standing, not fighting.

**What this run proves:** correct exploration (frontier-first, novelty-
aware, no false merges across 226 discovered rooms), correct shop detection
(2 real shops found and reported with full transcripts: "The Grunting
Boar" and "The General Store", live-verified in an earlier, uncontaminated
run before this session's Main Street corridor was discovered), correct
memory persistence and resume across process restarts, and correct
keyword-family matching machinery (never triggered falsely).

**What it doesn't yet prove:** finding this specific world's bakery, which
may simply be in a different, unexplored direction from spawn (the agent
never got there - it explored east into a corridor/wilderness/tower
sequence instead). A shorter or differently-anchored demo would either (a)
run with a bounded step budget from a fresh spawn and demonstrate the full
pipeline against whichever shop it finds first (already proven: General
Store, Grunting Boar), narrated as "the engine, not the specific target,
is the milestone-1 deliverable", or (b) manually steer the first few moves
away from the Main Street corridor (e.g. west/south from the Temple instead
of the path that leads east) before handing off to autonomous exploration,
so the frontier explorer's search space excludes the corridor and the tower
dead end entirely.

- **No stamina/rest management.** tbaMUD movement points deplete with
  travel and regenerate only while resting/sleeping over real-time game
  ticks (measured: ~1 point per 3 real seconds). The agent does not detect
  "You are too exhausted." and rest - a long exploration run can stall on
  this, indistinguishable from a real dead end unless the transcript is
  inspected. Worked around manually during testing (rest the character
  between runs); reasonable milestone-1 scope gap, not attempted in code.
- **No door-opening.** A closed exit shows as `(direction)` in tbaMUD's
  `Exits:` line; the agent does not recognize this notation specially or
  attempt `open <direction>` - it just tries the raw exit letter as given
  (including the parentheses), which fails and gets marked blocked. Same
  category of gap as stamina management - a real capability a fuller agent
  would need, out of scope for pure exploration in milestone 1.

## Running it

Needs its own venv (this project is the only pip dependency in the whole
graded repo - everything else, including journey_map, is stdlib-only):

```bash
cd week0_explore/challenge_agent
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

export ANTHROPIC_API_KEY=sk-...        # never commit this - read from env only
export MUD_USERNAME=yourcharactername  # required, no default - never commit a real one
export MUD_PASSWORD=yourpassword       # required, no default - never commit a real one

# journey_map's proxy must be running first (this agent connects through
# it, not directly to the MUD, so journey_map's viewer sees it explore too):
cd ../journey_map && python3 -u journey_map.py &
cd ../challenge_agent

.venv/bin/python challenge_agent.py "find the bakery"
```

Progress is saved continuously to `agent_memory.json` - if the process is
killed mid-exploration, running the same command again resumes instead of
restarting cold.

### Dev/testing bypass

`--skip-front-door find_place --keyword bakery` runs the deterministic
engine directly, without calling the LLM at all - useful for iterating on
the engine without spending API calls, or for environments without
`ANTHROPIC_API_KEY` set. This is never the path used for a real challenge;
it exists purely for testing the engine in isolation.

## Files

- `challenge_agent.py` - the deterministic execution engine: telnet client
  + login, room-block parsing, self-taught localizer, frontier explorer,
  shop/mob detector, `find_place` goal executor, BFS pathfinder, report
  generator, CLI entry point. Zero AI.
- `challenge_front_door.py` - the only file that imports `anthropic`. Both
  LLM call sites live here.
- `../mud_protocol.py` - shared telnet/ANSI protocol layer (IAC stripping,
  line buffering, room-block regex/color constants), used by both this
  agent and `journey_map/`. Extracted from `journey_map.py` on 2026-07-17 -
  pure protocol decoding, no world knowledge, no behavior change to
  journey_map. See that module's docstring for exactly what moved.
- `test_challenge_agent.py` - stdlib `unittest` coverage of the localizer's
  pure functions (no network/MUD/LLM needed). Run:
  `.venv/bin/python -m unittest test_challenge_agent.py -v`
- `requirements.txt` - just `anthropic`.
- `agent_memory.json` - gitignored, per-run state.
