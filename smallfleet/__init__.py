"""SmallFleet: agent-loop routing for agent tool calls.

The router lives INSIDE a LangGraph-style agent execution loop. It classifies
each individual LLM call the agent makes by kind (routing, extraction,
tool call, summarization, reasoning, writing) and sends it to the cheapest
model tier that can handle that call. The policy is learned from the agent's
own runs on its own task set.

This is agent-loop routing, not LLM routing. Generic prompt routers route
standalone prompts or chat requests; SmallFleet routes per tool call inside
a running agent. See README.md for the prior-art note.
"""

__version__ = "0.1.0"
