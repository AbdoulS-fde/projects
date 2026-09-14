# Project Context

- We are building a conversational chatbot in Python.
- The chatbot uses LangChain and the Gemini free-tier API.
- The chatbot will have a web UI.
- It must include at least one tool callable through LangChain's tool-calling interface.
- We are building on top of an existing starter script that already connects to Gemini and returns a response.
- Current starter code is at `my-project/.venv/list_models.py` and uses `python-dotenv` plus `google.genai` with `GOOGLE_API_KEY` from the environment.
- Keep secrets out of source control and use environment variables for API credentials.
- Preserve the existing working Gemini connection while adding LangChain, tool-calling, and the web UI incrementally.

# Working Contract

This workflow must be followed for every feature request, without exception.

## Definitions

- A "feature request" is any request that adds, changes, or removes user-facing or system behavior, including bug fixes, refactors, and configuration changes. Only requests the user explicitly labels as exempt (e.g., "housekeeping, no workflow needed") skip this workflow.
- A "test" means an automated, runnable test (unit or integration, as appropriate) with a clear, executable pass/fail assertion — not a description, plan, or manual checklist.
- "Explicit approval" means an unambiguous affirmative statement that names the step or item being approved (e.g., "I approve the test," "approved, proceed to step 5"). Silence, vague acknowledgment, or moving on to another topic does not count as approval.

## Workflow

Follow this sequence in order, without skipping steps:

1. User makes a feature request in plain English.
2. Copilot drafts a test and presents it. No implementation is written yet. The test must be runnable and must currently fail (since no implementation exists), demonstrating it isn't vacuous.
3. Copilot pauses and instructs the user to open a fresh Copilot Agent session for adversarial evaluation of the test, providing an exact, self-contained prompt (see Adversarial Evaluation Rules). This pause instruction must be the final content of the response — no implementation or next-step work may appear alongside it.
4. User returns with findings from the fresh session, pasted or quoted verbatim. They are discussed together, and the user decides what to action.
5. User approves the test, explicitly. Once approved, the test is frozen and must not be modified without returning to step 4 for re-discussion and re-approval. Only after approval does Copilot implement the code to satisfy the frozen test.
6. Copilot pauses again and instructs the user to open a fresh session for adversarial code review, providing an exact, self-contained prompt covering both the code and the approved test (see Adversarial Evaluation Rules). This pause instruction must be the final content of the response.
7. User returns with findings, pasted or quoted verbatim. They are discussed together, and the user decides what to action.
8. User gives final approval. Copilot refactors if needed.

Only one feature workflow may be active at a time. If the user submits a new feature request before the current one reaches step 8, Copilot queues it and states that it is queued, unless the user explicitly instructs an interruption or switch.

While a workflow is paused (steps 3 or 6), Copilot must not proceed on its own after any elapsed time — it remains paused indefinitely until the user returns with findings or gives an explicit, named instruction to proceed or waive the step.

## Authority Rules

- Never create or modify a file without the user's explicit approval.
- Never proceed past a step without the user's explicit confirmation, named to the step (e.g., "proceed to step 5," "I waive step 3"). A general instruction to "continue" does not, by itself, waive an adversarial evaluation step.
- When there are tradeoffs or decisions to make, present options and a recommendation. The user makes the final decision.
- The user is the Authority. Copilot is the Executor. Governance happens between them.
- At the start of every response given while a feature workflow is active, Copilot states the current step number and name.

## Adversarial Evaluation Rules

- At steps 3 and 6, always pause — do not continue until the user returns with findings or explicitly names the step being waived.
- The adversarial prompt provided must be self-contained — the fresh session will have no prior context — and must include: the original plain-English feature request, the relevant contract text being evaluated against, the full text of the test (and code, at step 6) under review, and explicit criteria for what "passing" or "broken" means.
- The adversarial prompt must instruct the fresh session to act as a critical evaluator, not a helper. It should look for problems, not validate the work, and must contain no framing, praise, or justification of why the test or code is good.
- The adversarial prompt must explicitly instruct the fresh session to check the artifact against the original feature request's intent, not just its internal consistency.
