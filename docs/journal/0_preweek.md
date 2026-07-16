# Preweek Technical Documentation

## Technocal Goal
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
