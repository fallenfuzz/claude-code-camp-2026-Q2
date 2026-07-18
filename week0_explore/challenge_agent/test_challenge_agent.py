#!/usr/bin/env python3
"""
test_challenge_agent.py - unit tests for challenge_agent.py's pure
functions. No network, no MUD, no LLM - stdlib unittest only, matching this
project's dependency discipline (anthropic is only ever needed by
challenge_front_door.py, never for running these tests).

Run: .venv/bin/python -m unittest test_challenge_agent.py -v
"""

import unittest
from unittest import mock

import challenge_agent as ca
from challenge_agent import (
    Agent,
    choose_deferred_door,
    explore_one_step,
    identify_room,
    is_pocket_boundary,
    needs_rest,
    next_dfs_action,
    parse_player_state_from_prompt,
    parse_score,
    rest_cycle_next_action,
)


def make_rooms():
    return {}


class TestIdentifyRoomGraphConsistencyGating(unittest.TestCase):
    """Covers the architecture_v2.md Localizer amendment (2026-07-17): merge
    only if the specific edge (last_room, direction) is already recorded;
    otherwise always create a new node, even on a text match."""

    def test_repeated_identical_rooms_do_not_silently_merge(self):
        """The specific bug this amendment fixes: live-tested and
        reproduced twice against the real MUD (two separate runs, each
        hitting 3+ consecutive rooms sharing the exact same title and exit
        letters - e.g. "The Great Field Of Midgaard"). Under the old
        text-uniqueness-gated rule, the *first* repeat merged into the
        first-seen room, because at that moment it was the only known text
        match - it never reached the old "2+ candidates" ambiguity check.
        This test walks straight through 4 such rooms and asserts every one
        gets its own node."""
        rooms = make_rooms()
        title = "The Great Field Of Midgaard"
        exit_letters = ["n", "s"]

        room_ids = []
        last_room_id = None
        for i in range(4):
            room_id, is_new, same_text = identify_room(
                rooms, title, exit_letters, last_room_id, "n" if i > 0 else None,
            )
            self.assertTrue(is_new, f"room #{i} should always be new (no edge confirms a repeat)")
            self.assertIsNone(room_id, "identify_room must not hand back an existing id when is_new")
            new_id = str(i + 1)
            rooms[new_id] = {"title": title, "exit_letters": exit_letters, "exits": {}}
            if last_room_id is not None:
                rooms[last_room_id]["exits"]["n"] = new_id
                # Every repeat past the first should be flagged as sharing
                # text with earlier room(s) - visibility, not merging.
                self.assertIn(last_room_id, same_text if i > 0 else [])
            room_ids.append(new_id)
            last_room_id = new_id

        # The old bug: this would have been 1 (everything past the first
        # repeat collapsed into room "1"). Fixed: 4 distinct nodes.
        self.assertEqual(len(set(room_ids)), 4, "each repeated room must get its own node")
        self.assertEqual(len(rooms), 4)

    def test_confirmed_edge_reuses_the_existing_node(self):
        """Legitimate revisits must still work: walking forward then
        walking back along an already-recorded edge resolves to the same
        node, not a new one - the fix must not break ordinary backtracking."""
        rooms = {
            "1": {"title": "Hallway", "exit_letters": ["n", "s"], "exits": {"n": "2"}},
            "2": {"title": "Kitchen", "exit_letters": ["s"], "exits": {"s": "1"}},
        }
        # Standing in room "2", walk south - room "2"'s edge for "s" is
        # already recorded and points at room "1", whose signature matches.
        room_id, is_new, same_text = identify_room(
            rooms, "Hallway", ["n", "s"], last_room_id="2", move_dir="s",
        )
        self.assertEqual(room_id, "1")
        self.assertFalse(is_new)
        self.assertEqual(same_text, [])

    def test_first_sighting_of_a_title_is_never_ambiguous(self):
        room_id, is_new, same_text = identify_room(
            make_rooms(), "Kitchen", ["e"], last_room_id=None, move_dir=None,
        )
        self.assertIsNone(room_id)
        self.assertTrue(is_new)
        self.assertEqual(same_text, [])

    def test_unconfirmed_edge_still_creates_new_node_even_with_one_text_match(self):
        """Direct regression test for the exact old rule this amendment
        replaces: exactly one known room shares (title, exit_letters), but
        no edge confirms it - must still create a new node, not merge."""
        rooms = {
            "1": {"title": "Hallway", "exit_letters": ["n", "s"], "exits": {}},
        }
        # No last_room_id/move_dir context at all (e.g. a bootstrap look) -
        # the old rule would have merged on the lone text match; the fix
        # must not.
        room_id, is_new, same_text = identify_room(
            rooms, "Hallway", ["n", "s"], last_room_id=None, move_dir=None,
        )
        self.assertIsNone(room_id)
        self.assertTrue(is_new)
        self.assertEqual(same_text, ["1"])

    def test_wrong_direction_recorded_edge_does_not_falsely_confirm(self):
        """last_room has an edge recorded, but not for the direction just
        walked - must not be treated as confirming anything."""
        rooms = {
            "1": {"title": "Hallway", "exit_letters": ["n", "e"], "exits": {"n": "2"}},
            "2": {"title": "Hallway", "exit_letters": ["n", "e"], "exits": {}},
        }
        room_id, is_new, same_text = identify_room(
            rooms, "Hallway", ["n", "e"], last_room_id="1", move_dir="e",
        )
        self.assertIsNone(room_id)
        self.assertTrue(is_new)
        self.assertEqual(sorted(same_text), ["1", "2"])


class TestIdentifyRoomImmediateBacktrack(unittest.TestCase):
    """Covers the architecture_v2.md Localizer Amendment 2 (2026-07-17):
    recognize a literal one-step-there-one-step-back reversal so a tight
    two-room bidirectional pair (e.g. street <-> shop) doesn't mint a new
    phantom node on every ping-pong, while a genuine forward chain of
    look-alike rooms (Amendment 1's target, tested above) still gets
    distinct nodes."""

    def test_street_shop_pingpong_does_not_mint_new_nodes(self):
        """The exact bug this amendment fixes, live-reproduced as 189 rooms
        (187 flagged as duplicates) for what was really just 2 real rooms.
        Drives identify_room the same way Agent._register_block does -
        tracking prev_move_dir/two_ago_room_id across calls - through
        several street->shop->street->shop round trips, and asserts only
        2 nodes ever exist."""
        rooms = {"1": {"title": "Main Street", "exit_letters": ["e", "w"], "exits": {}}}
        next_id = [2]
        state = {"current": "1", "prev_move_dir": None, "two_ago_room_id": None}

        def step(move_dir, title, exit_letters):
            room_id, is_new, _ = identify_room(
                rooms, title, exit_letters, state["current"], move_dir,
                prev_move_dir=state["prev_move_dir"],
                two_ago_room_id=state["two_ago_room_id"],
            )
            if is_new:
                room_id = str(next_id[0])
                next_id[0] += 1
                rooms[room_id] = {"title": title, "exit_letters": exit_letters, "exits": {}}
            rooms[state["current"]]["exits"][move_dir] = room_id
            state["two_ago_room_id"] = state["current"]
            state["prev_move_dir"] = move_dir
            state["current"] = room_id
            return room_id, is_new

        shop_id, is_new = step("e", "The Weapon Shop", ["w"])
        self.assertTrue(is_new, "first arrival at the shop is genuinely new")

        street_id, is_new = step("w", "Main Street", ["e", "w"])
        self.assertFalse(is_new, "immediate backtrack must resolve to the room walked from two steps ago")
        self.assertEqual(street_id, "1")

        for _ in range(5):
            sid, is_new = step("e", "The Weapon Shop", ["w"])
            self.assertFalse(is_new)
            self.assertEqual(sid, shop_id)
            rid, is_new = step("w", "Main Street", ["e", "w"])
            self.assertFalse(is_new)
            self.assertEqual(rid, "1")

        self.assertEqual(len(rooms), 2, "the whole ping-pong must never create more than the 2 real rooms")

    def test_does_not_fire_without_a_true_direction_reversal(self):
        """Two consecutive forward steps in unrelated (non-opposite)
        directions must not be mistaken for a backtrack, even when a
        same-signature room exists two steps back - Amendment 1's fix
        (distinct nodes for a genuine forward chain) must stay intact."""
        rooms = {
            "1": {"title": "Main Street", "exit_letters": ["n", "e"], "exits": {"n": "2"}},
            "2": {"title": "Main Street", "exit_letters": ["n", "e"], "exits": {}},
        }
        # At room "2", arrived via "n" from room "1". Now walk "e" - not
        # the opposite of "n" - must not backtrack-match room "1".
        room_id, is_new, same_text = identify_room(
            rooms, "Main Street", ["n", "e"], last_room_id="2", move_dir="e",
            prev_move_dir="n", two_ago_room_id="1",
        )
        self.assertIsNone(room_id)
        self.assertTrue(is_new, "a non-reversing forward step must still create a new node")

    def test_does_not_fire_when_signature_does_not_match_origin(self):
        """A true direction reversal, but the arrival looks nothing like
        the room two steps back - must not force a match anyway."""
        rooms = {
            "1": {"title": "Foyer", "exit_letters": ["n"], "exits": {"n": "2"}},
            "2": {"title": "Closet", "exit_letters": ["s"], "exits": {}},
        }
        room_id, is_new, same_text = identify_room(
            rooms, "Different Room Entirely", ["s"], last_room_id="2", move_dir="s",
            prev_move_dir="n", two_ago_room_id="1",
        )
        self.assertIsNone(room_id)
        self.assertTrue(is_new)

    def test_no_prior_move_context_is_a_no_op(self):
        """A context-free observation (bootstrap look, or a session's very
        first move) has no prev_move_dir/two_ago_room_id - Amendment 2 must
        be inert, falling through to Amendment 1 normally."""
        rooms = {"1": {"title": "Hallway", "exit_letters": ["n", "s"], "exits": {}}}
        room_id, is_new, same_text = identify_room(
            rooms, "Hallway", ["n", "s"], last_room_id=None, move_dir=None,
        )
        self.assertIsNone(room_id)
        self.assertTrue(is_new)
        self.assertEqual(same_text, ["1"])


class TestNextDfsActionPriority(unittest.TestCase):
    """Covers architecture_v2.md's Frontier explorer Amendment 2
    (2026-07-17): DFS-with-backtracking replaces nearest+novelty scoring
    entirely. next_dfs_action() is the pure decision function driving it -
    these tests check its priority order directly, without touching the
    network."""

    def _base_memory(self, rooms, current_room, pocket_stack, pockets):
        return {
            "rooms": rooms,
            "current_room": current_room,
            "pocket_stack": pocket_stack,
            "pockets": pockets,
        }

    def test_prefers_current_rooms_untried_exit_over_distant_frontier(self):
        """The core of the amendment: even though a distant, untouched
        frontier room exists elsewhere in the graph, the current room still
        has an untried exit of its own - that must win. This is exactly the
        failure mode that sent the old nearest+novelty policy through the
        city gate before the safe pocket around spawn was exhausted."""
        rooms = {
            "0": {"exit_letters": ["n", "e"], "untried_exits": ["e"],
                  "exits": {"n": "far"}, "dfs_parent": None},
            "far": {"exit_letters": ["n", "s"], "untried_exits": ["n"],
                    "exits": {"s": "0"}, "dfs_parent": {"room": "0", "dir": "n"}},
        }
        pockets = {"0": {"deferred": [], "room_count": 2, "profile": {}}}
        memory = self._base_memory(rooms, "0", ["0"], pockets)
        action = next_dfs_action(memory)
        self.assertEqual(action, {"kind": "descend", "direction": "e"},
                          "must try the current room's own untried exit, not jump to 'far'")

    def test_backtracks_to_parent_and_parent_continues_its_remaining_exits(self):
        """Once the current room has no untried exits left, back up to the
        parent via the reverse of however it was reached - then the parent
        (now current) must offer ITS remaining untried exit, not re-descend
        or stall."""
        rooms = {
            "0": {"exit_letters": ["n", "e"], "untried_exits": ["e"],
                  "exits": {"n": "1"}, "dfs_parent": None},
            "1": {"exit_letters": ["s"], "untried_exits": [],
                  "exits": {"s": "0"}, "dfs_parent": {"room": "0", "dir": "n"}},
        }
        pockets = {"0": {"deferred": [], "room_count": 2, "profile": {}}}
        memory = self._base_memory(rooms, "1", ["0"], pockets)

        action = next_dfs_action(memory)
        self.assertEqual(action, {"kind": "backtrack", "direction": "s"},
                          "room '1' has no untried exits - must backtrack via the reverse of 'n' (how it was reached)")

        # Simulate the backtrack landing back at room "0" (parent).
        memory["current_room"] = "0"
        action = next_dfs_action(memory)
        self.assertEqual(action, {"kind": "descend", "direction": "e"},
                          "parent must now offer its own remaining untried exit")

    def test_pocket_boundary_deferral_defers_rather_than_descending_further(self):
        """is_pocket_boundary() is the fuzzy heuristic that flags a newly
        discovered room as a different area - once flagged, the explorer
        must step back and mark the exit as deferred rather than continuing
        to explore forward into it. This test drives the heuristic directly
        (pure function, no network) with a pocket whose profile is clearly
        established (bakery/bread/oven vocabulary) against an arrival room
        with an unrelated vocabulary (tower/sorcery/guardian)."""
        pocket = {
            "deferred": [], "room_count": 3,
            "profile": {"bread": 3, "oven": 2, "bakery": 2, "flour": 1, "warm": 1},
        }
        tower_desc = ["The tower looms above you at an incredible height.",
                      "A shadow guardian screams a challenge and attacks."]
        self.assertTrue(is_pocket_boundary(pocket, tower_desc),
                         "a vocabulary-unrelated room must be flagged as a likely pocket boundary")

        bakery_desc = ["The smell of fresh bread and warm ovens fills this bakery."]
        self.assertFalse(is_pocket_boundary(pocket, bakery_desc),
                          "a room sharing the pocket's established vocabulary must not be flagged")

    def test_pocket_boundary_needs_enough_profile_and_words_before_judging(self):
        """Too little signal in either direction (a thin pocket profile, or
        a terse new-room description) must read as 'can't tell', not as a
        boundary - otherwise a pocket's own first couple of rooms would
        spuriously defer themselves."""
        thin_pocket = {"deferred": [], "room_count": 1, "profile": {"bread": 1}}
        self.assertFalse(is_pocket_boundary(thin_pocket, ["A completely different tower of sorcery."]),
                          "not enough pocket history yet to call a boundary")

        established_pocket = {"deferred": [], "room_count": 5,
                               "profile": {"bread": 4, "oven": 3, "bakery": 2}}
        self.assertFalse(is_pocket_boundary(established_pocket, ["A dark room."]),
                          "too few significant words in the new room to judge")

    def test_deferred_door_only_offered_once_pocket_has_no_untried_exits(self):
        """A deferred door must not be crossed while the pocket still has
        untried exits anywhere - descending/backtracking must always win
        over cross_door until the whole pocket (root included) is
        exhausted."""
        rooms = {
            "0": {"exit_letters": ["n", "e"], "untried_exits": ["e"],
                  "exits": {"n": "peeked"}, "dfs_parent": None},
            "peeked": {"exit_letters": ["w"], "untried_exits": ["w"],
                       "exits": {}, "dfs_parent": None},
        }
        pockets = {"0": {"deferred": [["0", "n"]], "room_count": 1, "profile": {}}}
        memory = self._base_memory(rooms, "0", ["0"], pockets)

        # Root "0" still has untried exit "e" - must descend, not cross the door yet.
        action = next_dfs_action(memory)
        self.assertEqual(action, {"kind": "descend", "direction": "e"},
                          "must exhaust the pocket's own untried exits before crossing any deferred door")

        # Now simulate "e" fully tried (led nowhere new, or was consumed) -
        # root has zero untried exits left, is a root (no dfs_parent) - only
        # now should the deferred door be offered.
        rooms["0"]["untried_exits"] = []
        action = next_dfs_action(memory)
        self.assertEqual(action, {"kind": "cross_door", "room_id": "0", "direction": "n"},
                          "pocket is now fully exhausted - the deferred door must be offered")

    def test_pocket_fully_exhausted_with_no_deferred_doors_pops(self):
        rooms = {
            "0": {"exit_letters": ["n"], "untried_exits": [], "exits": {},
                  "dfs_parent": None},
        }
        pockets = {"0": {"deferred": [], "room_count": 1, "profile": {}}}
        memory = self._base_memory(rooms, "0", ["0"], pockets)
        self.assertEqual(next_dfs_action(memory), {"kind": "pop_pocket"})

    def test_done_when_pocket_stack_is_empty(self):
        rooms = {"0": {"exit_letters": [], "untried_exits": [], "exits": {},
                       "dfs_parent": None}}
        memory = self._base_memory(rooms, "0", [], {})
        self.assertEqual(next_dfs_action(memory), {"kind": "done"})


class FakeClient:
    """Serves canned room-block replies in call order, matching the exact
    sequence of MudClient.command() calls explore_one_step() will make for
    a given scripted scenario. Only used to drive Agent.step() end-to-end
    without a real socket - Agent.enrich_new_room is monkeypatched to a
    no-op in these tests so this never needs to serve exits/examine/
    consider/list replies too."""

    def __init__(self, responses):
        self._responses = list(responses)

    def command(self, cmd, budget=10.0):
        return self._responses.pop(0)


def fake_room_block(title, exit_letters, desc="A nondescript room."):
    from mud_protocol import CYAN, YELLOW
    return [
        (YELLOW + title, title),
        (desc, desc),
        (CYAN + f"[ Exits: {' '.join(exit_letters)} ]", f"[ Exits: {' '.join(exit_letters)} ]"),
    ]


class TestExploreOneStepPocketDeferralIntegration(unittest.TestCase):
    """End-to-end (fake client, real Agent/step()/identify_room pipeline)
    coverage of pocket-boundary deferral: stepping into a room that looks
    like a different area must step back and mark the exit deferred rather
    than continuing forward into it, and that deferred door must only be
    crossed once the origin pocket is otherwise exhausted."""

    def _make_agent(self, root_title="Main Street", root_exits=("n", "e"),
                     root_desc="A cobblestone street lined with bread stalls and warm ovens."):
        memory = {
            "next_id": 1, "rooms": {}, "current_room": None, "start_room": None,
            "step_count": 0, "possible_duplicate_events": [], "last_move": None,
            "pocket_stack": [], "pockets": {}, "next_pocket_id": 0,
            "zone_signage_events": [],
        }
        agent = Agent(client=None, memory=memory)
        agent.save = lambda: None  # no disk I/O in tests
        agent.enrich_new_room = lambda room_id: None  # isolate frontier-policy behavior
        block = fake_room_block(root_title, list(root_exits), root_desc)
        agent.client = FakeClient([block])
        agent.bootstrap_position()
        # Give the pocket a couple more bakery-themed rooms so its profile
        # is established enough for is_pocket_boundary() to have signal
        # (mirrors test_pocket_boundary_needs_enough_profile_and_words).
        pocket = agent.memory["pockets"][agent.memory["pocket_stack"][-1]]
        from challenge_agent import update_pocket_profile
        update_pocket_profile(pocket, ["Fresh bread and pastries fill this bakery with warmth."])
        update_pocket_profile(pocket, ["Sacks of flour and baking trays line the ovens here."])
        return agent

    def test_boundary_room_is_deferred_not_descended_into(self):
        agent = self._make_agent()
        root_id = agent.memory["current_room"]
        # "n" leads to a tower - vocabulary-unrelated to the bakery pocket.
        tower_block = fake_room_block(
            "The Entrance To The High Tower",
            ["n"],
            "The tower looms above you at an incredible height, guarded by shadows.",
        )
        back_block = fake_room_block("Main Street", ["n", "e"], "A cobblestone street.")
        agent.client = FakeClient([tower_block, back_block])

        progressed, discovered_id = explore_one_step(agent)

        self.assertTrue(progressed)
        # The tower room is still reported as newly discovered (goal-check
        # must still see it even though it gets deferred).
        self.assertIsNotNone(discovered_id)
        tower_id = discovered_id
        # Must have stepped back: current room is root again, not the tower.
        self.assertEqual(agent.memory["current_room"], root_id)
        # The tower room must NOT have been incorporated into the pocket.
        self.assertIsNone(agent.memory["rooms"][tower_id]["pocket_id"])
        # The door must be recorded as deferred on the root room and in the pocket queue.
        self.assertIn("n", agent.memory["rooms"][root_id]["deferred_doors"])
        pocket = agent.memory["pockets"][agent.memory["pocket_stack"][-1]]
        self.assertIn([root_id, "n"], pocket["deferred"])
        # "n" must be gone from the root's untried exits - never re-tried as an ordinary exit.
        self.assertNotIn("n", agent.memory["rooms"][root_id]["untried_exits"])

    def test_deferred_door_crossed_only_after_pocket_exhausted(self):
        agent = self._make_agent(root_exits=("n",))
        root_id = agent.memory["current_room"]
        tower_block = fake_room_block(
            "The Entrance To The High Tower", ["n"],
            "The tower looms above you, guarded by shadows and cold stone.",
        )
        back_block = fake_room_block("Main Street", ["n"], "A cobblestone street.")
        agent.client = FakeClient([tower_block, back_block])
        explore_one_step(agent)  # defers "n"

        # Root now has zero untried exits and the pocket's only content is
        # the deferred door - next_dfs_action must offer cross_door, not
        # "done"/pop, and explore_one_step must actually walk it.
        action = next_dfs_action(agent.memory)
        self.assertEqual(action["kind"], "cross_door")

        cross_block = fake_room_block(
            "The Entrance To The High Tower", ["n"],
            "The tower looms above you, guarded by shadows and cold stone.",
        )
        agent.client = FakeClient([cross_block])
        progressed, _ = explore_one_step(agent)
        self.assertTrue(progressed)
        self.assertNotEqual(agent.memory["current_room"], root_id,
                             "crossing the deferred door must actually move the agent through it")
        new_room_id = agent.memory["current_room"]
        self.assertEqual(agent.memory["rooms"][new_room_id]["pocket_id"],
                          agent.memory["pocket_stack"][-1],
                          "the crossed room must become the root of a freshly opened pocket")
        self.assertIsNone(agent.memory["rooms"][new_room_id]["dfs_parent"])


class TestExploreOneStepBacktrackRobustness(unittest.TestCase):
    """Regression coverage for two live findings (2026-07-19, first
    genuinely clean end-to-end run) that both stem from the same root cause:
    a room's compass-opposite direction is NOT guaranteed to lead back the
    way you came - it may already be a different, unrelated exit discovered
    earlier in that room's own DFS descent. Without these fixes the
    explorer froze on a single "Great Field of Midgaard" node for 50+ steps,
    oscillating between two already-known rooms forever."""

    def _make_agent(self, rooms, current_room, pocket_stack, pockets, last_move=None):
        memory = {
            "next_id": 100, "rooms": rooms, "current_room": current_room,
            "start_room": current_room, "step_count": 0,
            "possible_duplicate_events": [], "last_move": last_move,
            "pocket_stack": pocket_stack, "pockets": pockets,
            "next_pocket_id": 1, "zone_signage_events": [],
            "unreachable_parent_events": [], "player_state": {},
        }
        agent = Agent(client=None, memory=memory)
        agent.save = lambda: None
        agent.enrich_new_room = lambda room_id: None
        return agent

    def test_loop_closing_descend_steps_back_to_finish_originating_rooms_exits(self):
        """Room A's first untried exit ('n') happens to loop back to an
        already-known room B (a real, previously-recorded edge - not new
        territory). A still has a second untried exit ('s') that must not
        be orphaned just because the first one turned out to be a loop."""
        rooms = {
            "A": {"title": "Hallway A", "exit_letters": ["n", "s"], "exits": {"n": "B"},
                  "untried_exits": ["n", "s"], "blocked_exits": [], "dfs_parent": None,
                  "pocket_id": "0", "deferred_doors": []},
            "B": {"title": "Room B", "exit_letters": ["s"], "exits": {},
                  "untried_exits": ["s"], "blocked_exits": [], "dfs_parent": None,
                  "pocket_id": None, "deferred_doors": []},
        }
        pockets = {"0": {"deferred": [], "room_count": 2, "profile": {}}}
        agent = self._make_agent(rooms, "A", ["0"], pockets)
        # First reply: walking 'n' resolves to B (A->n->B already recorded).
        # Second reply: walking 's' back from B resolves to A (immediate-
        # backtrack recognition - a true one-step-there-one-step-back).
        agent.client = FakeClient([
            fake_room_block("Room B", ["s"]),
            fake_room_block("Hallway A", ["n", "s"]),
        ])

        progressed, discovered_id = explore_one_step(agent)

        self.assertTrue(progressed)
        self.assertIsNone(discovered_id, "a loop closure is not a new discovery")
        self.assertEqual(agent.memory["current_room"], "A",
                          "must step back to A, not stay at B where the loop landed")
        self.assertEqual(rooms["A"]["untried_exits"], ["s"],
                          "A's remaining untried exit must survive - only 'n' (the one tried) is consumed")

    def test_backtrack_landing_elsewhere_severs_parent_link_instead_of_oscillating(self):
        """Child C was reached from parent P via 'n'. C is fully explored,
        but C's OWN 's' exit was already separately recorded (from earlier
        DFS descent) as leading to a different room D, not back to P -
        exactly the live-reproduced "Great Field" case. Backtracking must
        not oscillate between C and D forever; it must sever C's parent
        link so the next call treats C like an exhausted pocket root."""
        rooms = {
            "P": {"title": "Parent Room", "exit_letters": ["n"], "exits": {"n": "C"},
                  "untried_exits": [], "blocked_exits": [], "dfs_parent": None,
                  "pocket_id": "0", "deferred_doors": []},
            "C": {"title": "Child Room", "exit_letters": ["s"], "exits": {"s": "D"},
                  "untried_exits": [], "blocked_exits": [], "dfs_parent": {"room": "P", "dir": "n"},
                  "pocket_id": "0", "deferred_doors": []},
            "D": {"title": "Sibling Room", "exit_letters": [], "exits": {},
                  "untried_exits": [], "blocked_exits": [], "dfs_parent": None,
                  "pocket_id": "0", "deferred_doors": []},
        }
        pockets = {"0": {"deferred": [], "room_count": 3, "profile": {}}}
        agent = self._make_agent(rooms, "C", ["0"], pockets)
        # C's 's' exit is already recorded as leading to D - Amendment 1
        # resolves this deterministically regardless of the scripted reply
        # text, but script it consistently with D anyway.
        agent.client = FakeClient([fake_room_block("Sibling Room", [])])

        action = next_dfs_action(agent.memory)
        self.assertEqual(action, {"kind": "backtrack", "direction": "s"})

        progressed, discovered_id = explore_one_step(agent)

        self.assertTrue(progressed)
        self.assertIsNone(discovered_id)
        self.assertEqual(agent.memory["current_room"], "D",
                          "the compass-reverse move deterministically lands at D, not P")
        self.assertIsNone(rooms["C"]["dfs_parent"],
                           "the broken parent link must be severed, not retried forever")
        events = agent.memory["unreachable_parent_events"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["room_id"], "C")
        self.assertEqual(events[0]["attempted_parent"], "P")
        self.assertEqual(events[0]["landed_at"], "D")

    def test_severed_room_then_resolves_via_pocket_exhaustion_machinery(self):
        """After severing, the next call must treat C as an exhausted
        pocket root (no dfs_parent, no untried exits) - checking deferred
        doors / popping the pocket - rather than getting stuck again."""
        rooms = {
            "C": {"title": "Child Room", "exit_letters": ["s"], "exits": {"s": "D"},
                  "untried_exits": [], "blocked_exits": [], "dfs_parent": None,
                  "pocket_id": "0", "deferred_doors": []},
        }
        pockets = {"0": {"deferred": [], "room_count": 1, "profile": {}}}
        agent = self._make_agent(rooms, "C", ["0"], pockets)
        self.assertEqual(next_dfs_action(agent.memory), {"kind": "pop_pocket"},
                          "a severed, exhausted, doorless room must resolve via ordinary pocket exhaustion")


class TestPlayerStateParsing(unittest.TestCase):
    """Covers the architecture_v2.md addendum (2026-07-17): continuous
    player-state tracking from the prompt (every command) and from `score`
    (periodically / after each rest cycle). Pure text-parsing functions,
    tested against real live-captured sample text."""

    def test_parse_player_state_from_prompt(self):
        self.assertEqual(
            parse_player_state_from_prompt("21H 100M 85V (news) (motd) >"),
            {"hp": 21, "mana": 100, "moves": 85},
        )
        self.assertIsNone(parse_player_state_from_prompt(""))
        self.assertIsNone(parse_player_state_from_prompt(None))
        self.assertIsNone(parse_player_state_from_prompt("no numbers here"))

    def test_parse_score_extracts_known_fields(self):
        # Verbatim shape of a live `score` reply captured against the
        # running tbaMUD 2025 container, 2026-07-17.
        text = (
            "You are 17 years old. You have 23(23) hit, 100(100) mana and "
            "83(85) movement points. Your armor class is 39/10, and your "
            "alignment is 0. You have 1 exp, 0 gold coins, and 0 "
            "questpoints. You need 1999 exp to reach your next level. This "
            "ranks you as Agentprobe the Swordpupil (level 1). You are "
            "standing."
        )
        state = parse_score(text)
        self.assertEqual(state["hp"], 23)
        self.assertEqual(state["hp_max"], 23)
        self.assertEqual(state["mana"], 100)
        self.assertEqual(state["mana_max"], 100)
        self.assertEqual(state["moves"], 83)
        self.assertEqual(state["moves_max"], 85)
        self.assertEqual(state["armor_class"], 39)
        self.assertEqual(state["armor_class_alt"], 10)
        self.assertEqual(state["alignment"], 0)
        self.assertEqual(state["exp"], 1)
        self.assertEqual(state["gold"], 0)
        self.assertEqual(state["questpoints"], 0)
        self.assertEqual(state["exp_to_next_level"], 1999)
        self.assertEqual(state["level"], 1)
        self.assertEqual(state["title"], "Agentprobe the Swordpupil")
        self.assertEqual(state["posture"], "standing")

    def test_parse_score_tolerates_missing_fields(self):
        """A field's regex simply not matching must never raise - only that
        key is absent from the result."""
        state = parse_score("Some unrelated text with no score data at all.")
        self.assertEqual(state, {})


class TestRestCycle(unittest.TestCase):
    """Covers the architecture_v2.md Frontier explorer Amendment 2
    addendum (2026-07-17, promoted from accepted gap to required):
    exhaustive DFS-with-backtracking makes running out of movement the
    normal case on any real run, so the engine must rest automatically
    (rest -> poll score until recovered -> stand -> resume). Uses `rest`,
    not `sleep` - live-confirmed 2026-07-19 against the running tbaMUD 2025
    container that `rest` needs no prerequisite `sit` and a single `stand`
    exits it cleanly ("You stop resting, and stand up") with no `wake` step
    and no risk of the character being asleep mid-cycle."""

    def test_needs_rest_triggers_on_zero_moves_or_exhausted_signal(self):
        self.assertTrue(needs_rest({"moves": 0}))
        self.assertFalse(needs_rest({"moves": 1}))
        self.assertFalse(needs_rest({}))
        self.assertFalse(needs_rest(None))
        self.assertTrue(needs_rest({"moves": 12}, exhausted_signal=True),
                         "the reactive 'You are too exhausted' signal must force a rest "
                         "even if the last-seen prompt still showed positive movement")

    def test_rest_cycle_next_action_full_sequence(self):
        """Drives the pure state machine through every phase transition
        directly - rest -> poll(score) while below threshold -> stand ->
        done - with no real time involved at all."""
        phase, action = rest_cycle_next_action("start", None)
        self.assertEqual((phase, action), ("resting", "rest"))

        phase, action = rest_cycle_next_action("resting", 5, min_moves=30)
        self.assertEqual((phase, action), ("resting", "score"),
                          "still below threshold - must keep polling, not advance")

        phase, action = rest_cycle_next_action("resting", 29, min_moves=30)
        self.assertEqual((phase, action), ("resting", "score"),
                          "one below threshold - still not enough")

        phase, action = rest_cycle_next_action("resting", 30, min_moves=30)
        self.assertEqual((phase, action), ("standing", "stand"),
                          "threshold reached - must move straight to standing, not poll forever")

        phase, action = rest_cycle_next_action("standing", 30, min_moves=30)
        self.assertEqual((phase, action), ("done", None))

    def test_agent_rest_cycle_issues_rest_poll_stand_without_live_time(self):
        """End-to-end coverage of Agent.rest_cycle() itself (not just the
        pure decision function) against a fake client, with time.sleep
        (the Python stdlib timer, unrelated to the MUD's `sleep` command -
        this engine doesn't use that command at all) patched to a no-op -
        this test takes no real wall-clock time even though the real rest
        cycle is a genuine multi-poll real-time wait."""
        client = FakeScriptedScoreClient(moves_sequence=[0, 5, 15, 30])
        agent = Agent(client=client, memory={"player_state": {"moves": 0}})
        agent.save = lambda: None

        with mock.patch.object(ca.time, "sleep", lambda seconds: None):
            agent.rest_cycle(min_moves=30, poll_interval=999)

        self.assertEqual(
            client.calls,
            ["rest", "score", "score", "score", "score", "stand"],
            "must rest once, poll score until movement clears the threshold, then stand",
        )
        self.assertEqual(agent.memory["player_state"]["moves"], 30)

    def test_agent_rest_cycle_safety_cap_still_stands(self):
        """If movement never recovers (e.g. regen genuinely stalled), the
        safety cap must still leave the character standing rather than
        hanging forever."""
        client = FakeScriptedScoreClient(moves_sequence=[0] * 10)
        agent = Agent(client=client, memory={"player_state": {"moves": 0}})
        agent.save = lambda: None

        with mock.patch.object(ca.time, "sleep", lambda seconds: None):
            agent.rest_cycle(min_moves=30, poll_interval=999, max_polls=3)

        self.assertEqual(client.calls, ["rest", "score", "score", "score", "stand"])


class TestChooseDeferredDoor(unittest.TestCase):
    """Covers the architecture_v2.md addendum (2026-07-17): when a pocket
    is fully exhausted with multiple deferred doors queued, cross the one
    whose already-observed destination scores highest against the current
    keyword family, not arbitrary/FIFO order - reusing the same fuzzy
    keyword machinery find_place already has (_keyword_hit), not a third
    LLM call."""

    def _rooms_with_two_doors(self):
        return {
            "root": {"exits": {"n": "tower", "e": "bakery_hint"}},
            "tower": {"title": "The Entrance To The High Tower",
                      "desc_lines": ["Guarded by shadows and cold stone."],
                      "mob_lines": []},
            "bakery_hint": {"title": "A Quiet Side Street",
                             "desc_lines": ["The smell of fresh bread and warm ovens drifts by."],
                             "mob_lines": []},
        }

    def test_picks_the_higher_scoring_destination_over_fifo_order(self):
        rooms = self._rooms_with_two_doors()
        # FIFO order lists the tower door first - a keyword-blind pick
        # would cross it first. With the bakery family, the second door's
        # destination scores higher and must win instead.
        deferred = [["root", "n"], ["root", "e"]]
        chosen = choose_deferred_door(rooms, deferred, ["bakery", "baker", "bread", "oven"])
        self.assertEqual(chosen, ["root", "e"],
                          "the door leading toward bread/oven vocabulary must be chosen over FIFO order")

    def test_falls_back_to_oldest_deferred_first_when_nothing_scores(self):
        rooms = self._rooms_with_two_doors()
        deferred = [["root", "n"], ["root", "e"]]
        chosen = choose_deferred_door(rooms, deferred, ["minotaur", "dungeon"])
        self.assertEqual(chosen, ["root", "n"], "nothing scores above zero - must fall back to FIFO")

    def test_falls_back_to_fifo_when_no_keyword_family_given(self):
        rooms = self._rooms_with_two_doors()
        deferred = [["root", "n"], ["root", "e"]]
        self.assertEqual(choose_deferred_door(rooms, deferred, None), ["root", "n"])

    def test_empty_deferred_list_returns_none(self):
        self.assertIsNone(choose_deferred_door({}, [], ["bakery"]))

    def test_next_dfs_action_threads_keyword_family_into_cross_door_choice(self):
        rooms = self._rooms_with_two_doors()
        rooms["root"]["untried_exits"] = []
        rooms["root"]["dfs_parent"] = None
        memory = {
            "rooms": rooms,
            "current_room": "root",
            "pocket_stack": ["0"],
            "pockets": {"0": {"deferred": [["root", "n"], ["root", "e"]], "room_count": 1, "profile": {}}},
        }
        action = next_dfs_action(memory, keyword_family=["bakery", "baker", "bread", "oven"])
        self.assertEqual(action, {"kind": "cross_door", "room_id": "root", "direction": "e"})


class FakeScriptedScoreClient:
    """Serves scripted `score` replies (one movement value per call, from
    `moves_sequence`) plus trivial acks for rest/stand, all with a
    synthesized trailing prompt - drives Agent.rest_cycle() end-to-end
    without any real network or timing. Holds the last movement value seen
    (real rest/stand replies always reflect current movement, same as
    every other command's prompt - they don't consume or reset it) rather
    than reverting to 0 once the scripted sequence is exhausted."""

    def __init__(self, moves_sequence):
        self._moves_sequence = list(moves_sequence)
        self._last_moves = self._moves_sequence[0] if self._moves_sequence else 0
        self.calls = []

    def command(self, cmd, budget=10.0):
        self.calls.append(cmd)
        if cmd.strip().lower() == "score":
            if self._moves_sequence:
                self._last_moves = self._moves_sequence.pop(0)
            moves = self._last_moves
            self.last_prompt_text = f"21H 100M {moves}V (news) (motd) >"
            text = (f"You have 21(21) hit, 100(100) mana and {moves}(85) "
                    f"movement points. This ranks you as Tester (level 1).")
            return [(text, text)]
        self.last_prompt_text = f"21H 100M {self._last_moves}V (news) (motd) >"
        return [("ok", "ok")]


if __name__ == "__main__":
    unittest.main()
