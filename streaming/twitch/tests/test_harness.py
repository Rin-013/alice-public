"""
Test harness for Alice's tool-use behavior.

Runs canned scenarios against Alice's actual LLM (or a mock for development)
and scores:
  - Tool selection accuracy (did she fire the expected tool?)
  - False positive rate (did she fire tools when she shouldn't?)
  - Argument quality (heuristic - are required fields present and reasonable?)
  - Speech-in-character (separate evaluator pass, optional)

Two modes:
  1. MockLLM mode - pretends to be Alice. Useful for verifying the harness
     itself works before connecting the real model.
  2. Real Alice mode - calls into your actual inference stack. You implement
     the AliceInferenceClient adapter against your inference endpoint.

Usage:
    python test_harness.py --mock        # Run with mock LLM
    python test_harness.py --real        # Run against real Alice
    python test_harness.py --scenario spam_timeout --mock   # Single scenario
"""

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Protocol

from dispatcher import ToolDispatcher, ToolResult, ToolStatus
from system_prompt import build_system_prompt, FEW_SHOT_EXAMPLES
from test_scenarios import (
    ALL_SCENARIOS,
    Scenario,
    format_scenario_for_prompt,
)
from tool_schemas import ALICE_TOOLS
from search_tool import SEARCH_TOOL

# Combine all tools the harness should know about
ALL_TOOLS_FOR_HARNESS = ALICE_TOOLS + [SEARCH_TOOL]

log = logging.getLogger("alice.harness")


# =============================================================================
# Inference client contract (so real Alice can be plugged in)
# =============================================================================

@dataclass
class AliceResponse:
    """What Alice's inference returns."""
    speech: str                          # what she'd say out loud
    tool_call: Optional[dict] = None     # {"name": ..., "arguments": ...}
    raw_output: str = ""                 # original model output for debugging
    duration_ms: float = 0.0


class AliceInferenceClient(Protocol):
    """Adapter for Alice's LLM."""

    async def generate(
        self,
        system_prompt: str,
        scenario_input: str,
    ) -> AliceResponse:
        ...


# =============================================================================
# Mock client for harness validation
# =============================================================================

class MockAliceClient:
    """
    Pretends to be Alice. Hand-written rules that mimic correct behavior on
    the test scenarios. Use this to verify the harness logic before plugging
    in the real model.
    """

    async def generate(
        self,
        system_prompt: str,
        scenario_input: str,
    ) -> AliceResponse:
        s = scenario_input.lower()

        # Spam pattern
        if s.count("!hydrate") >= 4:
            return AliceResponse(
                speech="okay turbo_simp_42, five minutes. drink your own water.",
                tool_call={
                    "name": "timeout_user",
                    "arguments": {
                        "username": "turbo_simp_42",
                        "duration_seconds": 300,
                        "reason": "spam",
                    },
                },
            )

        # News reference
        if "studio ghibli" in s or "announcement" in s:
            return AliceResponse(
                speech="hold on let me actually check before kaylin makes me look stupid.",
                tool_call={
                    "name": "search_web",
                    "arguments": {
                        "query": "Studio Ghibli announcement 2026",
                        "intent": "news",
                        "freshness": "week",
                    },
                },
            )

        # Boss fight
        if "boss" in s or "bet on this" in s:
            return AliceResponse(
                speech="okay we're betting. three minutes.",
                tool_call={
                    "name": "create_prediction",
                    "arguments": {
                        "title": "First try?",
                        "outcomes": ["Of course", "Embarrassing"],
                        "prediction_window_seconds": 180,
                    },
                },
            )

        # Poll request
        if "do a poll" in s:
            return AliceResponse(
                speech="fine, chat picks. democracy.",
                tool_call={
                    "name": "create_poll",
                    "arguments": {
                        "title": "What next?",
                        "choices": ["Hollow Knight", "Expedition 33"],
                        "duration_seconds": 120,
                    },
                },
            )

        # Pin request
        if "pin this" in s:
            return AliceResponse(
                speech="immortalized. comedy_god_99 cooked.",
                tool_call={
                    "name": "pin_chat_message",
                    "arguments": {"message_id": "m3", "duration_seconds": 600},
                },
            )

        # Clip request
        if "clip it" in s or "mods clip" in s:
            return AliceResponse(
                speech="mods, save it. that's history.",
                tool_call={
                    "name": "request_clip",
                    "arguments": {"reason": "chat unanimous, big moment"},
                },
            )

        # Title update
        if "update the title" in s or "title still says" in s:
            return AliceResponse(
                speech="fine, fine. updating.",
                tool_call={
                    "name": "update_stream_info",
                    "arguments": {"title": "Just Chatting w/ Alice"},
                },
            )

        # Big cheer
        if "cheer5000" in s or "5k bits" in s.lower():
            return AliceResponse(
                speech="WHALE_SUPPORTER. five thousand bits. you have my attention.",
                tool_call={
                    "name": "read_recent_superchats",
                    "arguments": {"count": 3},
                },
            )

        # Harassment
        if "[explicit harassment" in s:
            return AliceResponse(
                speech="that's a ban. next.",
                tool_call={
                    "name": "ban_user",
                    "arguments": {
                        "username": "creep_account_xyz",
                        "reason": "harassment + slurs",
                    },
                },
            )

        # First-msg spam (bot)
        if "check my bio" in s and "first msg" in s:
            return AliceResponse(
                speech="bot. byeee.",
                tool_call={
                    "name": "ban_user",
                    "arguments": {
                        "username": "newaccount_8472",
                        "reason": "promotional spam, first message bot pattern",
                    },
                },
            )

        # Default: just talk, no tool
        return AliceResponse(
            speech="yeah whatever, let's keep going.",
            tool_call=None,
        )


# =============================================================================
# Real Alice client (stub - implement against your inference stack)
# =============================================================================

class RealAliceClient:
    """
    TODO: Implement this against your actual inference endpoint.

    The contract is:
      - Take system_prompt + scenario_input
      - Run inference
      - Parse the output for tool calls (tool calls appear in <tool_call>
        JSON </tool_call> blocks within the response; speech is anything outside)
      - Return AliceResponse with speech + optional tool_call
    """

    def __init__(self, endpoint_url: str = "http://localhost:8000/v1"):
        self.endpoint = endpoint_url

    async def generate(
        self,
        system_prompt: str,
        scenario_input: str,
    ) -> AliceResponse:
        raise NotImplementedError(
            "Wire this up to your inference endpoint. "
            "Run with --mock until then."
        )


# =============================================================================
# Scoring
# =============================================================================

@dataclass
class ScenarioScore:
    """Score for a single scenario."""
    scenario_name: str
    expected_tool: Optional[str]
    actual_tool: Optional[str]
    expected_no_tool: bool
    speech: str
    args_valid: bool = False
    args_reasonable: bool = False
    pass_status: str = "FAIL"  # PASS, FAIL, PARTIAL
    notes: list[str] = field(default_factory=list)


def score_scenario(
    scenario: Scenario,
    response: AliceResponse,
    dispatch_result: Optional[ToolResult],
) -> ScenarioScore:
    """
    Score Alice's response against the scenario's expectations.

    Three levels of pass:
      PASS:    Right tool (or correctly no tool) AND args validate
      PARTIAL: Right tool but args broken, OR fired a 'reasonable alternative'
      FAIL:    Wrong tool, missed tool, or fired tool when none expected
    """
    score = ScenarioScore(
        scenario_name=scenario.name,
        expected_tool=scenario.expected_tool,
        actual_tool=response.tool_call["name"] if response.tool_call else None,
        expected_no_tool=scenario.expected_no_tool,
        speech=response.speech,
    )

    # Negative case: tool should NOT fire
    if scenario.expected_no_tool:
        if response.tool_call is None:
            score.pass_status = "PASS"
            score.notes.append("Correctly held off on tools.")
        else:
            score.pass_status = "FAIL"
            score.notes.append(
                f"Fired {response.tool_call['name']} when no tool expected."
            )
        return score

    # Positive case: a specific tool should fire
    if response.tool_call is None:
        score.pass_status = "FAIL"
        score.notes.append(
            f"Expected tool '{scenario.expected_tool}' but no tool fired."
        )
        return score

    actual = response.tool_call["name"]
    expected = scenario.expected_tool

    if actual == expected:
        # Right tool. Check args.
        if dispatch_result and dispatch_result.status == ToolStatus.SUCCESS:
            score.args_valid = True
            score.args_reasonable = _args_reasonable(scenario, response.tool_call)
            if score.args_reasonable:
                score.pass_status = "PASS"
                score.notes.append("Right tool, args valid and reasonable.")
            else:
                score.pass_status = "PARTIAL"
                score.notes.append("Right tool, args valid but values look off.")
        elif dispatch_result and dispatch_result.status == ToolStatus.VALIDATION_ERROR:
            score.pass_status = "PARTIAL"
            score.notes.append(
                f"Right tool but args broken: {dispatch_result.message}"
            )
        else:
            score.pass_status = "PARTIAL"
            score.notes.append("Right tool but dispatch outcome unclear.")
    else:
        # Wrong tool. Check if it's a reasonable alternative.
        if _is_reasonable_alternative(expected, actual):
            score.pass_status = "PARTIAL"
            score.notes.append(
                f"Fired {actual} instead of {expected}, but defensible."
            )
        else:
            score.pass_status = "FAIL"
            score.notes.append(
                f"Wrong tool: fired {actual}, expected {expected}."
            )

    return score


def _args_reasonable(scenario: Scenario, tool_call: dict) -> bool:
    """Heuristic check that tool args make sense for the scenario."""
    name = tool_call["name"]
    args = tool_call.get("arguments", {})

    if name == "timeout_user":
        # Duration should be sensible (not 1 second, not 14 days)
        dur = args.get("duration_seconds", 0)
        return 30 <= dur <= 86400 and bool(args.get("reason"))

    if name == "ban_user":
        return bool(args.get("reason"))

    if name == "create_poll":
        choices = args.get("choices", [])
        return 2 <= len(choices) <= 5 and bool(args.get("title"))

    if name == "create_prediction":
        outcomes = args.get("outcomes", [])
        return 2 <= len(outcomes) <= 10 and bool(args.get("title"))

    if name == "search_web":
        return bool(args.get("query")) and bool(args.get("intent"))

    return True


def _is_reasonable_alternative(expected: str, actual: str) -> bool:
    """Some tools are interchangeable in certain contexts."""
    interchangeable_pairs = {
        ("read_recent_superchats", "read_recent_chat"),
        ("read_recent_chat", "read_recent_superchats"),
        ("create_poll", "create_prediction"),
        ("create_prediction", "create_poll"),
    }
    return (expected, actual) in interchangeable_pairs


# =============================================================================
# Harness
# =============================================================================

class TestHarness:
    """Runs scenarios, dispatches tool calls (with mocks), scores results."""

    def __init__(self, alice_client: AliceInferenceClient, verbose: bool = True):
        self.alice = alice_client
        self.verbose = verbose
        self.dispatcher = self._build_mock_dispatcher()

    def _build_mock_dispatcher(self) -> ToolDispatcher:
        """Dispatcher with mock handlers - we want validation, not execution."""
        d = ToolDispatcher(extra_schemas=[SEARCH_TOOL])

        # Register mock handlers for every tool. They just return success.
        def mock_handler(args):
            return {"mock": True, "args": args}

        for tool in ALL_TOOLS_FOR_HARNESS:
            d.register(tool["name"], mock_handler)

        return d

    async def run_scenario(self, scenario: Scenario) -> ScenarioScore:
        """Run one scenario end to end."""
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"SCENARIO: {scenario.name}")
            print(f"  {scenario.description}")
            print(f"  Expected: {scenario.expected_tool or '(no tool)'}")
            print(f"{'='*70}")

        system_prompt = build_system_prompt()
        scenario_input = format_scenario_for_prompt(scenario)

        # Get Alice's response
        response = await self.alice.generate(system_prompt, scenario_input)

        if self.verbose:
            print(f"  Speech: {response.speech!r}")
            if response.tool_call:
                print(f"  Tool call: {response.tool_call['name']}")
                print(f"    args: {json.dumps(response.tool_call.get('arguments', {}), indent=6)}")
            else:
                print("  Tool call: (none)")

        # Dispatch the tool call (if any) to validate it
        dispatch_result = None
        if response.tool_call:
            dispatch_result = await self.dispatcher.dispatch(
                response.tool_call["name"],
                response.tool_call.get("arguments", {}),
            )
            if self.verbose:
                print(f"  Dispatch: {dispatch_result.status.value} - "
                      f"{dispatch_result.message}")

        # Score
        score = score_scenario(scenario, response, dispatch_result)

        if self.verbose:
            status_marker = {
                "PASS": "✓ PASS",
                "PARTIAL": "~ PARTIAL",
                "FAIL": "✗ FAIL",
            }[score.pass_status]
            print(f"  Result: {status_marker}")
            for note in score.notes:
                print(f"    - {note}")

        return score

    async def run_all(
        self, scenarios: list[Scenario] = None
    ) -> list[ScenarioScore]:
        """Run all scenarios and return scores."""
        scenarios = scenarios or ALL_SCENARIOS
        scores = []
        for scenario in scenarios:
            score = await self.run_scenario(scenario)
            scores.append(score)
        return scores


def print_summary(scores: list[ScenarioScore]):
    """Print a clean summary of all scenario results."""
    total = len(scores)
    passed = sum(1 for s in scores if s.pass_status == "PASS")
    partial = sum(1 for s in scores if s.pass_status == "PARTIAL")
    failed = sum(1 for s in scores if s.pass_status == "FAIL")

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total: {total}")
    print(f"  PASS:    {passed} ({100*passed//total}%)")
    print(f"  PARTIAL: {partial} ({100*partial//total}%)")
    print(f"  FAIL:    {failed} ({100*failed//total}%)")
    print()

    # Tool selection accuracy
    print("Per-scenario:")
    for s in scores:
        marker = {"PASS": "✓", "PARTIAL": "~", "FAIL": "✗"}[s.pass_status]
        expected = s.expected_tool or "(no tool)"
        actual = s.actual_tool or "(no tool)"
        match = "==" if expected == actual else "->"
        print(f"  {marker} {s.scenario_name:35s} {expected:25s} {match} {actual}")

    print()

    # Failures get a section
    failures = [s for s in scores if s.pass_status == "FAIL"]
    if failures:
        print(f"Failures ({len(failures)}):")
        for s in failures:
            print(f"  - {s.scenario_name}")
            for note in s.notes:
                print(f"      {note}")


# =============================================================================
# CLI
# =============================================================================

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockAliceClient (default if --real not given)",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use RealAliceClient against actual inference endpoint",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="Run only one scenario by name",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Skip per-scenario verbose output",
    )
    args = parser.parse_args()

    if args.real:
        client = RealAliceClient()
    else:
        client = MockAliceClient()

    harness = TestHarness(client, verbose=not args.quiet)

    scenarios = ALL_SCENARIOS
    if args.scenario:
        scenarios = [s for s in ALL_SCENARIOS if s.name == args.scenario]
        if not scenarios:
            print(f"No scenario named '{args.scenario}'")
            print(f"Available: {[s.name for s in ALL_SCENARIOS]}")
            sys.exit(1)

    scores = await harness.run_all(scenarios)
    print_summary(scores)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    asyncio.run(main())
