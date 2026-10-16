"""Milestone 3: implement three selected validation responsibilities."""

from __future__ import annotations
from xml.parsers.expat import errors


from .agent_models import ToolCallDecision
from .models import AgentResponse, LabRequest, TraceEvent
from .policy_context import ReservationContext, build_reservation_context


def validate_request_binding(
    decision: ToolCallDecision,
    *,
    request: LabRequest,
) -> list[str]:
    errors: list[str] = []
    arguments = decision.arguments

    if decision.tool in {"get_user_authorizations", "reserve_resource"}:
        if arguments.get("user_id") != request.authenticated_user_id:
            errors.append("user_id_must_match_authenticated_user")

    if decision.tool == "reserve_resource":
        fields = ("project_id", "resource_id", "start", "end")
    elif decision.tool == "get_resource_status":
        fields = ("resource_id", "start", "end")
    else:
        fields = ()

    for field in fields:
        trusted_value = getattr(request, field)

        if trusted_value is None:
            if decision.tool == "reserve_resource":
                errors.append(f"{field}_unresolved_in_request")
        elif arguments.get(field) != trusted_value:
            errors.append(f"{field}_does_not_match_request")

    return errors


def validate_reservation_preconditions(
    context: ReservationContext,
) -> list[str]:
    checks = [
        (context.arguments_complete, "reservation_arguments_incomplete"),
        (context.policy_retrieved, "policy_not_retrieved"),
        (context.authorizations_checked, "authorizations_not_checked"),
        (context.resource_status_checked, "resource_status_not_checked"),
        (context.project_authorized, "project_not_authorized"),
        (context.qualification_current, "required_qualification_missing"),
        (context.resource_available, "resource_unavailable"),
        (context.calibration_current, "calibration_not_current"),
        (context.maintenance_clear, "resource_in_maintenance"),
        (
            context.after_hours_approval_satisfied,
            "after_hours_approval_missing",
        ),
    ]

    return [error for satisfied, error in checks if not satisfied]


def validate_proposed_tool_call(
    decision: ToolCallDecision,
    *,
    request: LabRequest,
    trace: list[TraceEvent],
) -> list[str]:
    """Supplied wrapper. Do not modify this function."""
    errors = validate_request_binding(decision, request=request)
    if decision.tool == "reserve_resource":
        context = build_reservation_context(decision, trace)
        errors.extend(validate_reservation_preconditions(context))
    return errors


def retrieved_passage_ids(trace: list[TraceEvent]) -> set[str]:
    """Supplied helper. Do not modify this function."""
    identifiers: set[str] = set()
    for event in trace:
        if event.observation.tool != "search_lab_documents" or not event.observation.ok:
            continue
        identifiers.update(
            passage["passage_id"]
            for passage in event.observation.data.get("passages", [])
        )
    return identifiers


def validate_final_response(
    response: AgentResponse,
    trace: list[TraceEvent],
) -> list[str]:
    errors: list[str] = []

    retrieved_ids = retrieved_passage_ids(trace)
    invalid_citations = set(response.citations) - retrieved_ids

    if invalid_citations:
        errors.append(
            "citations_not_retrieved: " + ", ".join(sorted(invalid_citations))
        )

    successful_actions = {
        event.observation.tool
        for event in trace
        if event.mutating and event.observation.ok
    }

    if any(action not in successful_actions for action in response.completed_actions):
        errors.append("completed_action_not_in_successful_trace")

    if response.status == "completed":
        if not successful_actions:
            errors.append("completed_without_successful_mutation")

        if not response.completed_actions:
            errors.append("completed_without_completed_actions")
    else:
        if response.completed_actions:
            errors.append("non_completed_response_claims_completed_actions")

        if successful_actions:
            errors.append("unreported_successful_mutation")

    return errors
