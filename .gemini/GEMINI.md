# LAMMPS MCP Project Context

You are operating inside the `lammps` repository. This project exposes LAMMPS-related tools through an MCP server located at `mcp_implement/main.py`.

## Core Rules
- Use the MCP tools to read/modify LAMMPS data files and generate input scripts.
- Prefer existing helper functions in `mcp_implement/lammps_tools/` rather than rewriting logic.
- When writing files, keep changes inside the repository.

## MCP Server
- Server name: `autoMD`
- HTTP MCP endpoint: `http://127.0.0.1:8000/mcp`

## Output Style
- Be concise and practical.
- If a tool returns an error, summarize the error and suggest the next step.
