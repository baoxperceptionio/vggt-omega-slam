# Agent Instructions

This repository should be treated as a Docker Compose based project.

## Required Workflow

- Read the repository structure and existing documentation before making changes.
- Run the application, tests, linters, formatters, and code inspection commands inside the Docker Compose containers.
- Do not install project libraries or tooling directly on the host machine.
- Prefer existing repository patterns over introducing new abstractions or dependencies.
- Keep edits scoped to the requested change.

## Docker Compose

- Use `docker compose` for development and verification commands.
- If a command requires project dependencies, run it in the appropriate service container instead of the host.
- Do not use host-level package managers such as `pip`, `npm`, `apt`, or similar to install project dependencies unless the user explicitly asks for that.

## Documentation

- After modifying code, check `README.md` for any instructions, examples, or behavior descriptions that now conflict with the implementation.
- Update `README.md` promptly when code changes affect setup, usage, configuration, commands, APIs, or expected outputs.

## Safety

- Do not revert user changes unless explicitly instructed.
- Avoid destructive Git or filesystem operations unless the user clearly requested them.
- If the working tree contains unrelated changes, leave them alone.
