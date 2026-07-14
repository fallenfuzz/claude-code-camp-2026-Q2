1. An agent file with ferenced files eg. Agent.nd, @~/docs/*.MD

haiku

Observations:
- Coding Harness will read local files, not pertatiing to the loop, it will take it off task and waste tokens/usage
- The agent ended up creating temp files tocreate a socket connection and execute commands, we should be persisting a common interface for the mud eg. mud_manager
- When it creates a rigid script and fails to login, it starts going off task looking for config files, its obvious that its scripts to login and interface in flawed, a mud_manager would remove this obstacle for small models