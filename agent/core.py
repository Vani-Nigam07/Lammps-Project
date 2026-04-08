# agent/core.py
import json
import os

import anthropic

from .tools import TOOL_DEFINITIONS, dispatch_tool

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:  # Optional dependency
    genai = None
    genai_types = None

_ANTHROPIC_CLIENT = anthropic.Anthropic()

class LammpsAgent:
    def __init__(self, data, run_config, status_callback=None):
        self.data = data                   # parsed LAMMPS data dict from pore_editor
        self.run_config = run_config       # {pressure, steps, seed, output_name}
        self.status_callback = status_callback  # fn(msg) → pushes to Streamlit
        self.messages = []
        self.provider = self._select_provider()
        self.gemini_client = self._init_gemini_client() if self.provider == "gemini" else None
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")

    def _select_provider(self) -> str:
        explicit = os.getenv("LAMMPS_AGENT_PROVIDER", "").strip().lower()
        if explicit in {"gemini", "anthropic"}:
            return explicit
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_USE_VERTEXAI"):
            return "gemini"
        return "anthropic"

    def _init_gemini_client(self):
        if genai is None:
            raise RuntimeError(
                "Gemini provider selected but google-genai is not installed. "
                "Install with: pip install google-genai"
            )
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            return genai.Client(api_key=api_key)
        return genai.Client()

    def _gemini_tools(self):
        function_declarations = []
        for tool in TOOL_DEFINITIONS:
            function_declarations.append(
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                }
            )
        return genai_types.Tool(function_declarations=function_declarations)

    def _gemini_extract_calls(self, response):
        calls = []
        if getattr(response, "function_calls", None):
            calls = list(response.function_calls)
        elif response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if getattr(part, "function_call", None):
                    calls.append(part)
        return calls

    def _gemini_call_name_args(self, call_part):
        if getattr(call_part, "name", None) and getattr(call_part, "args", None):
            return call_part.name, call_part.args
        if getattr(call_part, "function_call", None):
            fc = call_part.function_call
            return fc.name, fc.args
        raise RuntimeError("Unrecognized Gemini function call shape.")

    def run(self):
        """Main agentic loop: plan → validate → write → execute → parse → report."""
        initial_prompt = f"""
You are a LAMMPS simulation agent. You have been given a graphene pore membrane
data file and run configuration. Your job is to:
1. Validate the data file for consistency
2. Write the data file and input script to the working directory
3. Run LAMMPS in the bash sandbox
4. Monitor and stream the log output
5. Parse thermodynamic results
6. Report success or diagnose errors and suggest fixes

Run config: {json.dumps(self.run_config)}
Atom count: {self.data['counts']['atoms']}
Filter atoms (type 2): {sum(1 for a in self.data['atoms'] if a['type'] == 2)}
Piston atoms (type 1): {sum(1 for a in self.data['atoms'] if a['type'] == 1)}
"""

        if self.provider == "gemini":
            self._run_gemini(initial_prompt)
            return

        self.messages = [{
            "role": "user",
            "content": initial_prompt
        }]

        while True:
            response = _ANTHROPIC_CLIENT.messages.create(
                model="claude-opus-4-6",
                max_tokens=4096,
                tools=TOOL_DEFINITIONS,
                messages=self.messages,
            )

            # Append assistant turn
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                # Extract final text report
                for block in response.content:
                    if block.type == "text":
                        if self.status_callback:
                            self.status_callback(block.text)
                break

            # Handle tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    if self.status_callback:
                        self.status_callback(f"⚙️ Agent calling: `{block.name}`")
                    result = dispatch_tool(block.name, block.input, self.data)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    })

            if tool_results:
                self.messages.append({"role": "user", "content": tool_results})

    def _run_gemini(self, initial_prompt: str) -> None:
        tool = self._gemini_tools()
        config = genai_types.GenerateContentConfig(tools=[tool])
        contents = [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text(text=initial_prompt)],
            )
        ]

        while True:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=contents,
                config=config,
            )

            if response.candidates and response.candidates[0].content:
                contents.append(response.candidates[0].content)

            function_calls = self._gemini_extract_calls(response)
            if not function_calls:
                if self.status_callback and getattr(response, "text", None):
                    self.status_callback(response.text)
                break

            for call_part in function_calls:
                name, args = self._gemini_call_name_args(call_part)
                if self.status_callback:
                    self.status_callback(f"⚙️ Agent calling: `{name}`")
                try:
                    result = dispatch_tool(name, args, self.data)
                    if isinstance(result, dict) and "error" in result:
                        function_response = {"error": result["error"]}
                    else:
                        function_response = {"result": result}
                except Exception as exc:
                    function_response = {"error": str(exc)}

                function_response_part = genai_types.Part.from_function_response(
                    name=name,
                    response=function_response,
                )
                contents.append(
                    genai_types.Content(role="tool", parts=[function_response_part])
                )
