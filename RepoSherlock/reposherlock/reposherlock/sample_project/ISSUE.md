# Documented example issue (for RepoSherlock demos)

Running the sample app fails at startup with:

    KeyError: 'db_url'

**Root cause (intentional):** `main.build_config()` stores the setting under the key
`"database_url"`, but `database.connect_database()` reads `config["db_url"]`.

**Good demo question:**

    python main.py ask sample_project "Which functions might be responsible for KeyError: 'db_url'?"

A good answer points at `connect_database` (where the key is read) and `build_config`
(where the config is built), cites their line numbers, and labels the key mismatch
as a hypothesis unless the evidence shows both lines.

**Also in this sample:** `connect_database` and `retry_connect` call each other, which
exercises the call-graph cycle handling.

This file is documentation only. RepoSherlock never runs the sample code.
