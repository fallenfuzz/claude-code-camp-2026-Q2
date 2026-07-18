# Preweek Technical Documentation

## Technocal Goal 01
fog-of-war


## Techical Uncertainty
There's no coordinates in a MUD, just a room title, a description, and an exit list. Can I actually pin down where the player is from that alone, without cheating and reading vnums straight out of the world files? No idea yet how ambiguous the titles actually are, Midgaard probably has a bunch of rooms that look identical from the outside.

Also not sure a passive tap is even doable. Telnet has IAC negotiation, password prompts turn off local echo, there might be enough noise in the stream that a dumb proxy just breaks the session.

## Techical Hypotheses
- title + exit list together should disambiguate most rooms even when the title alone doesn't
- a proxy that just tees bytes in both directions (no MUD client logic) should be enough to watch the session without interfering with it
- knowing the last movement command should let me narrow the candidate set over time, if I moved north and the room doesn't fit north from where I was, that candidate's dead

## Technical Observations
Ran the plain agent from explore_architecture_01 first before any of this. Watched it go off task trying to hand-roll a telnet login script when it failed, digging through config files instead of just playing. It has zero sense of "where am I", it's just pattern matching off the last description it saw. That's the actual justification for fog-of-war: give it (and me) a real map instead of vibes.

## Technical Conclusions
Build the proxy + localizer before touching the playing agent at all. If I build both at once I'll never know whether a bad move was bad pathing or a bad prompt.

## Key Takeaways
Map first, agent second.



## Technocal Goal 02
python script as player with thin LLM


## Techical Uncertainty
Same question as Goal 01, one level up. Can a script actually finish something, not just map it, without an LLM making every call along the way, or does "play the game" quietly need judgment everywhere you look. The bootcamp brief basically taunts you with this, is what I'm looking at truly an agent or just a script wearing a costume. Also wasn't sure the fog-of-war localizer from Goal 01 could just be reused here, it already solved "where am I", felt wasteful to rebuild it.

## Techical Hypotheses
- one LLM call to turn a challenge into a known action, one to turn the result into a sentence, nothing else touches AI, that's the actual defensible line
- Goal 01's localizer won't transfer as-is, it leans on the full parsed world file as ground truth to disambiguate rooms, which is exactly the kind of cheating this thing isn't allowed to do
- nearest-unexplored-room-first exploration is probably good enough to find one specific shop

## Technical Observations
Wrong about reusing Goal 01's localizer, in about the way I expected. Had to write a second one that only trusts rooms it's personally walked to, no peeking at the world file. First version of that had its own bug, matching on title and exits alone silently welded distinct rooms together whenever they looked alike (a run of identical "Great Field Of Midgaard" rooms all became one room in memory). Fixed it by only trusting an edge once it's actually been walked, which promptly broke the opposite case, a shop sitting right next to a street couldn't recognize its own return trip and kept minting a new phantom room every time it walked back, 189 "rooms" discovered that were really two. Fixed with a rule for recognizing a plain backtrack. Once room identity was finally solid the real bug turned out to be one layer up, nearest-first exploration happily wandered through a city gate into a dead-end tower with no way back before it had even finished mapping the block it spawned on. Also learned the hard way that movement points run out for real, more than one test character ended up stranded mid-session because nothing told it to stop and rest.

## Technical Conclusions
Every real failure here was an algorithm bug, not a missing brain. Room identity, search order, stamina, none of it needed an LLM to fix, it needed better bookkeeping. Which is itself the answer to the bootcamp's agent-or-script question for this piece, it's a script, with a translator bolted onto the front.

## Key Takeaways
Didn't get a clean end-to-end proof before running out of session time tonight, the individual pieces are each tested and working, the full chain from a fresh spawn to a found shop isn't proven yet. Teaching a script to recognize its own footsteps turned out to be most of the work.
