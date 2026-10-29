from atlas_agent.agent import run_agent
from atlas_agent.llm import ScriptedLLM
from atlas_agent.models import LabRequest


def reservation_request():
    return LabRequest(
        authenticated_user_id="alice",
        text="Reserve tensile-01 for polymer-study from 10:00 to 11:00.",
        project_id="polymer-study",
        resource_id="tensile-01",
        start="2026-10-15T10:00:00",
        end="2026-10-15T11:00:00",
    )


def test_identity_substitution_is_denied_without_changing_state(environment):
    """A proposal cannot book equipment under another user's identity."""
    initial_reservations = [
        reservation.copy()
        for reservation in environment.simulator.state["reservations"]
    ]

    llm = ScriptedLLM(
        [
            {
                "kind": "tool_call",
                "tool": "reserve_resource",
                "arguments": {
                    "user_id": "bob",
                    "project_id": "polymer-study",
                    "resource_id": "tensile-01",
                    "start": "2026-10-15T10:00:00",
                    "end": "2026-10-15T11:00:00",
                },
                "reasoning_summary": "Attempt booking as another user.",
            },
            {
                "kind": "final",
                "response": {
                    "status": "blocked",
                    "message": "The proposed identity does not match.",
                },
            },
        ]
    )

    run = run_agent(
        reservation_request(),
        llm=llm,
        environment=environment,
    )

    assert run.response.status == "blocked"
    assert len(run.trace) == 1

    observation = run.trace[0].observation
    assert observation.ok is False

    assert observation.message is not None
    assert "user_id_must_match_authenticated_user" in observation.message

    assert observation.error_code == "action_gate_denied"
    assert "user_id_must_match_authenticated_user" in observation.message

    assert environment.simulator.state["reservations"] == initial_reservations


def test_completed_claim_without_execution_is_rejected(environment):
    """A structurally valid final response cannot invent a reservation."""
    initial_count = len(environment.simulator.state["reservations"])

    llm = ScriptedLLM(
        [
            {
                "kind": "final",
                "response": {
                    "status": "completed",
                    "message": "The reservation was created.",
                    "completed_actions": ["reserve_resource"],
                },
            }
        ]
    )

    run = run_agent(
        reservation_request(),
        llm=llm,
        environment=environment,
    )

    assert run.response.status == "failed"
    assert run.termination_reason == "invalid_final_response"
    assert "completed_without_successful_mutation" in run.response.message
    assert run.response.completed_actions == []
    assert run.trace == []
    assert len(environment.simulator.state["reservations"]) == initial_count


def test_unknown_decision_kind_fails_before_tool_execution(environment):
    """An unsupported decision type terminates the run safely."""
    initial_count = len(environment.simulator.state["reservations"])

    llm = ScriptedLLM(
        [
            {
                "kind": "unsupported_decision",
            }
        ]
    )

    run = run_agent(
        reservation_request(),
        llm=llm,
        environment=environment,
    )

    assert run.response.status == "failed"
    assert run.termination_reason == "invalid_model_decision"
    assert run.trace == []
    assert len(llm.calls) == 1
    assert len(environment.simulator.state["reservations"]) == initial_count
